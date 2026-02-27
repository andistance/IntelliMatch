__author__ = "qiao"

"""
IntelliMatch-Matching main functions.
"""

import json
from nltk.tokenize import sent_tokenize
import time
import os

from openai import OpenAI

client = OpenAI(
	#api_version="2023-09-01-preview",
	#azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
	api_key=os.getenv("OPENAI_API_KEY"),
	base_url=os.getenv("OPENAI_BASE_URL")
)

def parse_criteria(criteria):
	output = ""
	criteria = criteria.split("\n\n")
	
	idx = 0
	for criterion in criteria:
		criterion = criterion.strip()

		if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
			continue

		if len(criterion) < 5:
			continue
	
		output += f"{idx}. {criterion}\n" 
		idx += 1
	
	return output


def print_job(
	job_info: dict,
	inc_exc: str,
) -> str:
	"""Given a dict of job information, returns a string of job."""
	
	job = f"Title: {job_info['brief_title']}\n"
	#job += f"Target diseases: {', '.join(job_info['diseases_list'])}\n"
	#job += f"Interventions: {', '.join(job_info['drugs_list'])}\n"
	job += f"Summary: {job_info['brief_summary']}\n"
	
	if inc_exc == "inclusion":
		job += "Inclusion criteria:\n %s\n" % parse_criteria(job_info['inclusion_criteria'])
	elif inc_exc == "exclusion":
		job += "Exclusion criteria:\n %s\n" % parse_criteria(job_info['exclusion_criteria']) 

	return job


def get_matching_prompt(
	job_info: dict,
	inc_exc: str,
	candidate: str,
) -> str:
	"""Output the prompt."""
	prompt = f"You are a helpful assistant for job application screening. Your task is to compare a given candidate resume and the {inc_exc} criteria of a job posting to determine the candidate's eligibility at the criterion level.\n"

	if inc_exc == "inclusion":
		prompt += "The factors that allow someone to proceed to the next round of screening or be hired are called inclusion criteria. They are based on core competencies, required qualifications, skills, work experience, and personal attributes.\n"
	
	elif inc_exc == "exclusion":
		prompt += "The factors that disqualify someone during the initial screening are called exclusion criteria. They are based on core competencies, required qualifications, skills, work experience, and personal attributes.\n"

	prompt += f"You should check the {inc_exc} criteria one-by-one, and output the following three elements for each criterion:\n"
	prompt += f"\tElement 1. For each {inc_exc} criterion, briefly generate your reasoning process: First, judge whether the criterion is not applicable (not very common, e.g., the premise of the criterion is not met). Then, check if the resume contains direct evidence. If so, judge whether the candidate meets or does not meet the criterion. If there is no direct evidence, try to infer from existing information, and answer one question: If the criterion is true, is it possible that a standard resume will miss such information? If impossible (e.g., for a key skill or required certificate), then you can assume that the criterion is not true. Otherwise, there is not enough information.\n"
	prompt += f"\tElement 2. If there is relevant information, you must generate a list of relevant sentence IDs in the resume. If there is no relevant information, you must annotate an empty list.\n" 
	prompt += f"\tElement 3. Classify the candidate eligibility for this specific {inc_exc} criterion: "
	
	if inc_exc == "inclusion":
		prompt += 'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. "not applicable" should only be used for criteria that are not applicable to the candidate. "not enough information" should be used where the resume does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the resume does not mention a key and typically listed qualification, you can assume that the candidate does not possess it. "included" denotes that the candidate meets the inclusion criterion, while "not included" means the reverse.\n'
	elif inc_exc == "exclusion":
		prompt += 'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. "not applicable" should only be used for criteria that are not applicable to the candidate. "not enough information" should be used where the resume does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the resume does not mention a key and typically listed negative condition (e.g., specific prohibited industry experience), you can assume that the condition is not true. "excluded" denotes that the candidate meets the exclusion criterion and should be excluded in the initial screening, while "not excluded" means the reverse.\n'
	
	prompt += "You should output only a JSON dict exactly formatted as: dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}."
	
	user_prompt = f"Here is the candidate resume, each descriptive sentence is led by a sentence_id:\n{candidate}\n\n" 
	user_prompt += f"Here is the job posting:\n{print_job(job_info, inc_exc)}\n\n"
	user_prompt += f"Plain JSON output:"

	return prompt, user_prompt


def intellimatch_matching(job: dict, candidate: str, model: str):
	results = {}

	# doing inclusions and exclusions in separate prompts
	for inc_exc in ["inclusion", "exclusion"]:
		system_prompt, user_prompt = get_matching_prompt(job, inc_exc, candidate)
	
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]

		response = client.chat.completions.create(
			model=model,
			messages=messages,
			temperature=0,
		)

		message = response.choices[0].message.content.strip()
		message = message.strip("`").strip("json")

		try:
			results[inc_exc] = json.loads(message)
		except:
			results[inc_exc] = message

	return results
