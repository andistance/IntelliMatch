__author__ = "qiao"

"""
Conduct the first stage retrieval by the hybrid retriever 
"""

from beir.datasets.data_loader import GenericDataLoader
import faiss
import json
from nltk import word_tokenize
import numpy as np
import os
from rank_bm25 import BM25Okapi
import sys
import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

def get_bm25_corpus_index(corpus):
	corpus_path = os.path.join(f"retrieval/bm25_corpus_{corpus}.json")

	# if already cached then load, otherwise build
	if os.path.exists(corpus_path):
		corpus_data = json.load(open(corpus_path))
		tokenized_corpus = corpus_data["tokenized_corpus"]
		corpus_nctids = corpus_data["corpus_nctids"]

	else:
		tokenized_corpus = []
		corpus_nctids = []

		with open(f"dataset/{corpus}/corpus_200_shard0.jsonl", "r", encoding='utf-8') as f:
			for line in f.readlines():
				entry = json.loads(line)
				corpus_nctids.append(entry["_id"])
				
				# weighting: 3 * title, 2 * condition, 1 * text
				tokens = word_tokenize(entry["title"].lower()) * 3 # word_tokenize() split all tokens in a string and put it in a list, e.g. "Diabetes Mellitus" -> ["Diabetes", "Mellitus"]
				#for disease in entry["metadata"]["diseases_list"]:
				#	tokens += word_tokenize(disease.lower()) * 2
				tokens += word_tokenize(entry["text"].lower())

				tokenized_corpus.append(tokens)

		corpus_data = {
			"tokenized_corpus": tokenized_corpus,
			"corpus_nctids": corpus_nctids,
		}

		with open(corpus_path, "w") as f:
			json.dump(corpus_data, f, indent=4)
	
	bm25 = BM25Okapi(tokenized_corpus) # class which build bm25 index

	return bm25, corpus_nctids

			
def get_bge_corpus_index(corpus):
	corpus_path = f"retrieval/{corpus}_embeds.npy" # cache the numpy array of the corpus embeddings
	nctids_path = f"retrieval/{corpus}_nctids.json" # cache the id of the corpus entries

	# if already cached then load, otherwise build
	if os.path.exists(corpus_path):
		embeds = np.load(corpus_path)
		corpus_nctids = json.load(open(nctids_path)) 

	else:
		embeds = []
		corpus_nctids = []

		model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").to("cuda")
		tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

		with open(f"dataset/{corpus}/corpus_200_shard0.jsonl", "r", encoding='utf-8') as f:
			print("Encoding the corpus")
			for line in tqdm.tqdm(f.readlines()):
				entry = json.loads(line)
				corpus_nctids.append(entry["_id"])

				title = entry["title"]
				text = entry["text"]

				with torch.no_grad():
					# tokenize the articles
					encoded = tokenizer(
						[[title, text]], 
						truncation=True, 
						padding=True, 
						return_tensors='pt', 
						max_length=512,
					).to("cuda")
					
					embed = model(**encoded).last_hidden_state[:, 0, :]

					embeds.append(embed[0].cpu().numpy())

		embeds = np.array(embeds)

		np.save(corpus_path, embeds)
		with open(nctids_path, "w") as f:
			json.dump(corpus_nctids, f, indent=4)

	index = faiss.IndexFlatIP(768)
	index.add(embeds)
	
	return index, corpus_nctids
	

if __name__ == "__main__":
	# different corpora, "trec_2021", "trec_2022", "sigir"
	corpus = sys.argv[1]

	# query type
	q_type = sys.argv[2]

	# different k for fusion
	k = int(sys.argv[3])

	# bm25 weight 
	bm25_wt = int(sys.argv[4])

	# bge weight
	bge_wt = int(sys.argv[5])

	# how many to rank for retriever
	N = 2000 

	# loading the qrels for only calculating the recall of our retriever, the qrels is the candidate-job ground truth mapping
	#_, _, qrels = GenericDataLoader(data_folder=f"dataset/{corpus}/").load(split="test")

	# loading all types of queries (load a mapping dictionary that associates each query ID with multiple formatted versions of the query, such as the original text, human-written summaries, AI-generated conditions, and clinician-authored terms)
	id2queries = json.load(open(f"dataset/{corpus}/id2queries_100_shard0.json", "r", encoding="utf-8"))

	# loading the indices
	print("Loading the bm25 indices ...")
	bm25, bm25_nctids = get_bm25_corpus_index(corpus)
	print("Finished, loading the bge indices ...")
	bge, bge_nctids = get_bge_corpus_index(corpus)
	print("Finished!")

	# loading the query encoder for bge
	model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").to("cuda")
	tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
	
	# then conduct the searches, saving top 1k
	output_path = f"results/{corpus}/qid2nctids_results_{q_type}_{corpus}_k{k}_bm25wt{bm25_wt}_bgewt{bge_wt}_N{N}.json"
	
	qid2nctids = {} # a mapping dictionary that associates each query ID with a list of retrieved clinical job IDs
	recalls = [] # a list of recall for each query

	with open(f"dataset/{corpus}/queries_100_shard0.jsonl", "r", encoding='utf-8') as f:
		for line in tqdm.tqdm(f.readlines()):
			entry = json.loads(line)
			query = entry["text"]
			qid = entry["_id"]

			#if qid not in qrels:
			#	continue
#
			#truth_sum = sum(qrels[qid].values())
			
			# get the keyword list
			if q_type in ["raw", "human_summary"]:
				conditions = [id2queries[qid][q_type]]
			elif "turbo" in q_type:
				conditions = id2queries[qid][q_type]["conditions"]
			elif "Clinician" in q_type:
				conditions = id2queries[qid].get(q_type, [])

			if len(conditions) == 0:
				nctid2score = {}
			else:
				# a list of nctid lists for the bm25 retriever
				bm25_condition_top_nctids = []

				for condition in conditions:
					tokens = word_tokenize(condition.lower())
					top_nctids = bm25.get_top_n(tokens, bm25_nctids, n=N)
					bm25_condition_top_nctids.append(top_nctids)
				
				# doing bge retrieval
				with torch.no_grad():
					encoded = tokenizer(
						conditions, 
						truncation=True, 
						padding=True, 
						return_tensors='pt', 
						max_length=256,
					).to("cuda")

					# encode the queries (use the [CLS] last hidden states as the representations)
					embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy() # (B, seq_len, hidden_size)

					# search the Faiss index, scores is lists of the similairties of top-N most similar jobs for all conditions, inds is lists of the indices of top-N most similar jobs for all conditions
					scores, inds = bge.search(embeds, k=N) # (C=len(conditions), N)

				bge_condition_top_nctids = []
				for ind_list in inds:
					top_nctids = [bge_nctids[ind] for ind in ind_list]
					bge_condition_top_nctids.append(top_nctids)

				nctid2score = {} # a mapping dictionary that associates each clinical job ID with the final hybrid score

				for condition_idx, (bm25_top_nctids, bge_top_nctids) in enumerate(zip(bm25_condition_top_nctids, bge_condition_top_nctids)):

					if bm25_wt > 0:
						for rank, nctid in enumerate(bm25_top_nctids):
							if nctid not in nctid2score:
								nctid2score[nctid] = 0
							
							nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))
					
					if bge_wt > 0:
						for rank, nctid in enumerate(bge_top_nctids):
							if nctid not in nctid2score:
								nctid2score[nctid] = 0
							
							nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))

			nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
			top_nctids = [nctid for nctid, _ in nctid2score[:N]]
			qid2nctids[qid] = top_nctids

			#actual_sum = sum([qrels[qid].get(nctid, 0) for nctid in top_nctids])
			#recalls.append(actual_sum / truth_sum)
	
	with open(output_path, "w") as f:
		json.dump(qid2nctids, f, indent=4)
