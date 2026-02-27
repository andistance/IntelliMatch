__author__ = "qiao"

"""
Rank the jobs given the matching and aggregation results
"""

import json
import sys

eps = 1e-9

def get_matching_score(matching):
	# count only the valid ones
	included = 0
	not_inc = 0
	na_inc = 0
	no_info_inc = 0

	excluded = 0
	not_exc = 0
	na_exc = 0
	no_info_exc = 0
	
	# first count inclusions
	for criteria, info in matching["inclusion"].items():
		
		if len(info) != 3:
			continue

		if info[2] == "included":
			included += 1	
		elif info[2] == "not included":
			not_inc += 1
		elif info[2] == "not applicable":
			na_inc += 1
		elif info[2] == "not enough information":
			no_info_inc += 1
	
	# then count exclusions
	for criteria, info in matching["exclusion"].items():

		if len(info) != 3:
			continue

		if info[2] == "excluded":
			excluded += 1	
		elif info[2] == "not excluded":
			not_exc += 1
		elif info[2] == "not applicable":
			na_exc += 1
		elif info[2] == "not enough information":
			no_info_exc += 1

	# get the matching score
	score = 0
	
	score += included / (included + not_inc + no_info_inc + eps)
	
	if not_inc > 0:
		score -= 1
	
	if excluded > 0:
		score -= 1
	
	return score 


def get_agg_score(assessment):
	try:
		rel_score = float(assessment["relevance_score_R"])
		eli_score = float(assessment["eligibility_score_E"])
	except:
		rel_score = 0
		eli_score = 0
	
	score = (rel_score + eli_score) / 100

	return score 


if __name__ == "__main__":
	# args are the results paths
	matching_results_path = sys.argv[1]
	agg_results_path = sys.argv[2]
	final_results_path = "results/job/final_results_job_gpt-4-turbo.json"

	# loading the results
	matching_results = json.load(open(matching_results_path))
	agg_results = json.load(open(agg_results_path))
	
	final_results = {}
	# loop over the candidates
	for candidate_id, label2job2results in matching_results.items():

		job2score = {}

		for _, job2results in label2job2results.items():
			for job_id, results in job2results.items():

				matching_score = get_matching_score(results)
				
				if candidate_id not in agg_results or job_id not in agg_results[candidate_id]:
					print(f"candidate {candidate_id} job {job_id} not in the aggregation results.")
					agg_score = 0
				else:
					agg_score = get_agg_score(agg_results[candidate_id][job_id])

				job_score = matching_score + agg_score
				
				job2score[job_id] = job_score

		sorted_job2score = sorted(job2score.items(), key=lambda x: -x[1])
		
		final_results[candidate_id] = sorted_job2score
		#print()
		#print(f"candidate ID: {candidate_id}")
		#print("Clinical job ranking:")
		#
		#for job, score in sorted_job2score:
		#	print(job, score)
#
		#print("===")
		#print()
	
	# save the final results
	json.dump(final_results, open(final_results_path, "w"), indent=4)
