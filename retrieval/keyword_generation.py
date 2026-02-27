__author__ = "qiao"

"""
generate the search keywords for each candidate
"""

import json
import os
from openai import OpenAI
from tqdm import tqdm

import sys

client = OpenAI(
	#api_version="2023-09-01-preview",
	#azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
	api_key=os.getenv("OPENAI_API_KEY"), # sk-mvaCtcK2Nelz4MQo8UUdTVLVhcDNUYkZz3ideenmHXH24zFG
	base_url=os.getenv("OPENAI_BASE_URL") # http://172.93.101.143:3001/v1
)


def get_keyword_generation_messages(note):
	#system = 'You are a helpful assistant and your task is to help search relevant clinical jobs for a given candidate #description. Please first summarize the main medical problems of the candidate. Then generate up to 32 key conditions for #searching relevant clinical jobs for this candidate. The key condition list should be ranked by priority. Please output #only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'
#
	#prompt =  f"Here is the candidate description: \n{note}\n\nJSON output:"

	system = 'You are a helpful assistant and your task is to help search relevant job positions for a given candidate\'s resume. Please first summarize the main professional profile of the candidate (including their key skills, experience level, industry background, and career highlights). Then generate up to 32 key search terms/conditions for finding relevant job positions that match this candidate. The key terms list should be ranked by priority (most important qualifications first). These terms can include required skills, experience years, education level, job titles, industries, or any other criteria that would help match this candidate to suitable positions. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'

	prompt = f"Here is the candidate's resume: \n{note}\n\nJSON output:"

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]
	
	return messages


if __name__ == "__main__":
	# the corpus: trec_2021, trec_2022, or sigir
	corpus = sys.argv[1]

	# the model index to use
	model = sys.argv[2]

	outputs = {}
	
	with open(f"dataset/{corpus}/queries.jsonl", "r", encoding='utf-8') as f:
		for line in tqdm(f.readlines()):
			entry = json.loads(line)
			messages = get_keyword_generation_messages(entry["text"])

			response = client.chat.completions.create(
				model=model,
				messages=messages,
				temperature=0,
			)

			output = response.choices[0].message.content
			output = output.strip("`").strip("json")
			
			outputs[entry["_id"]] = json.loads(output)

			with open(f"results/{corpus}/retrieval_keywords_shard0_{model}_{corpus}.json", "w") as f:
				json.dump(outputs, f, indent=4)
