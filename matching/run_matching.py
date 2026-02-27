__author__ = "qiao"

"""
Running the IntelliMatch matching.
"""

import json
from nltk.tokenize import sent_tokenize
import os
import sys
from tqdm import tqdm

from IntelliMatch import intellimatch_matching 

if __name__ == "__main__":
	corpus = sys.argv[1]
	model = sys.argv[2] 
	
	dataset = json.load(open(f"dataset/{corpus}/retrieved_jobs_100_shard0.json", "r", encoding="utf-8"))

	output_path = f"results/{corpus}/matching_results_{corpus}_{model}.json" 

	# Dict{Str(candidate_id): Dict{Str(label): Dict{Str(job_id): Str(output)}}}
	if os.path.exists(output_path):
		output = json.load(open(output_path))
	else:
		output = {}

	for instance in tqdm(dataset, total=len(dataset), desc=f"Processing {corpus} with {model} ..."):
		# Dict{'candidate': Str(candidate), '0': Str(NCTID), ...}
		candidate_id = instance["candidate_id"]
		candidate = instance["candidate"]
		sents = sent_tokenize(candidate)
		sents.append("The candidate will provide informed consent, and will comply with the company policies and job requirements without any practical issues.")
		sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
		candidate = "\n".join(sents)

		# initialize the candidate id in the output 
		if candidate_id not in output:
			output[candidate_id] = {"0": {}, "1": {}, "2": {}}
		
		for label in tqdm(["2", "1", "0"], total=len(["2", "1", "0"]), desc=f"Processing candidate {candidate_id} ..."):
			if label not in instance: continue

			for job in tqdm(instance[label], total=len(instance[label]), desc=f"Processing label {label} ..."): 
				job_id = job["NCTID"]

				# already calculated and cached
				if job_id in output[candidate_id][label]:
					continue
				
				# in case anything goes wrong (e.g., API calling errors)
				try:
					results = intellimatch_matching(job, candidate, model)
					output[candidate_id][label][job_id] = results

					with open(output_path, "w") as f:
						json.dump(output, f, indent=4)

				except Exception as e:
					print(e)
					continue
