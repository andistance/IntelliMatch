__author__ = "qiao"

"""
Using GPT to aggregate the scores by itself.
"""

from beir.datasets.data_loader import GenericDataLoader
import json
from nltk.tokenize import sent_tokenize
import os
import sys
import time
from tqdm import tqdm

from IntelliMatch import intellimatch_aggregation

if __name__ == "__main__":
	corpus = sys.argv[1] 
	model = sys.argv[2]

	# the path of the matching results
	matching_results_path = sys.argv[3]
	results = json.load(open(matching_results_path))

	# loading the job2info dict
	#job2info = json.load(open("dataset/job_info.json"))
	jobs_jsonl_file = f"dataset/{corpus}/corpus.jsonl"
	job2info = {}
    
	with open(jobs_jsonl_file, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line:
				try:
					job_data = json.loads(line)
					job_id = job_data.get('_id')
					cleaned_data = {k: v for k, v in job_data.get('metadata', {}).items()}
					cleaned_data['NCTID'] = job_id
					if job_id is not None:
						job2info[job_id] = cleaned_data
				except json.JSONDecodeError as e:
					print(f"警告：跳过无效的JSONL行: {e}")
	
	# loading the candidate info from dataset/{corpus}/queries.jsonl, return a dict which keys are candidate ids, values are candidate texts
	_, queries, _ = GenericDataLoader(data_folder=f"dataset/{corpus}/").load(split="test")
	
	# output file path
	output_path = f"results/{corpus}/aggregation_results_{corpus}_{model}.json"

	if os.path.exists(output_path):
		output = json.load(open(output_path))
	else:
		output = {}

	# candidate-level
	for candidate_id, info in tqdm(results.items(), total=len(results), desc="Aggregating results ..."):
		# get the candidate note
		candidate = queries[candidate_id]
		sents = sent_tokenize(candidate)
		sents.append("The candidate will provide informed consent, and will comply with the job protocol without any practical issues.")
		sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
		candidate = "\n".join(sents)

		if candidate_id not in output:
			output[candidate_id] = {}
		
		# label-level, 3 label / candidate
		for label, jobs in tqdm(info.items(), total=len(info), desc=f"Processing candidate {candidate_id} ..."):
				
			# job-level
			for job_id, job_results in tqdm(jobs.items(), total=len(jobs), desc=f"Processing label {label} ..."):
				# already cached results
				if job_id in output[candidate_id]:
					continue

				if type(job_results) is not dict:
					output[candidate_id][job_id] = "matching result error"

					with open(output_path, "w") as f:
						json.dump(output, f, indent=4)

					continue

				# specific job information
				job_info = job2info[job_id]	

				try:
					result = intellimatch_aggregation(candidate, job_results, job_info, model)
					output[candidate_id][job_id] = result 

					with open(output_path, "w") as f:
						json.dump(output, f, indent=4)

				except:
					continue
