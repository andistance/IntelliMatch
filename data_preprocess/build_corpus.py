import pandas as pd
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

# 初始化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

def generate_gpt_content(job_role, job_description, field_type):
    """生成GPT内容"""
    prompts = {
        "brief_summary": f"You are a summarization assistant. Your task is to generate a concise, single-paragraph summary of a given job description, based on its provided title and full text. The summary must be under 50 words, capture the role's core purpose and key responsibilities, and be written in clear, professional English. \n\n Job Title: \n {job_role} \n\n Job Description: \n {job_description} \n\n Your Summary: \n",
        "inclusion_criteria": f"You are a hiring assistant. Based on the provided Job Title and Job Description, generate 3 to 6 essential, specific, and verifiable inclusion criteria for candidates. Focus on must-have qualifications, skills, experiences, and attributes. Present each criterion as a **short**, **single**, and clear sentence. Do not number the items. Separate each criterion with two newlines (\n\n). \n\n Job Title: \n {job_role} \n\n Job Description: \n {job_description} \n\n Your Output: \n",
        "exclusion_criteria": f"You are a hiring screening assistant. Based on the provided Job Title and Job Description, generate 2 to 5 specific and verifiable exclusion criteria that would make a candidate ineligible for this role. Focus on clear disqualifiers, such as the absence of mandatory qualifications, possession of conflicting attributes, or a clear mismatch with core job requirements. Present each criterion as a **short**, **single**, and clear sentence. Do not number the items. Separate each criterion with two newlines (\n\n). \n\n Job Title: \n {job_role} \n\n Job Description: \n {job_description} \n\n Your Output: \n"
    }
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompts[field_type]}],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()

def csv_to_jsonl(csv_path, jsonl_path):
    """主函数"""
    # 读取CSV
    df = pd.read_csv(csv_path)
    df = df[:200]
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing csv ..."):
            # 基本字段
            data = {
                "_id": f"NCT0215{i+1}",
                "title": row['Job Roles'],
                "text": row['Job Description'],
                "metadata": {
                    "brief_title": row['Job Roles']
                }
            }
            
            # GPT生成字段
            try:
                data['metadata']['brief_summary'] = generate_gpt_content(row['Job Roles'], row['Job Description'], "brief_summary")
                time.sleep(1)
                
                data['metadata']['inclusion_criteria'] = generate_gpt_content(row['Job Roles'], row['Job Description'], "inclusion_criteria")
                time.sleep(1)
                
                data['metadata']['exclusion_criteria'] = generate_gpt_content(row['Job Roles'], row['Job Description'], "exclusion_criteria")
                time.sleep(1)
                
            except Exception as e:
                data.update({
                    'brief_summary': '',
                    'inclusion_criteria': '',
                    'exclusion_criteria': ''
                })
            
            # 写入
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            if (i+1) % 50 == 0:
                print(f"已处理: {i+1}/{len(df)} - {row['Job Roles'][:20]}")

# 运行
if __name__ == "__main__":
    csv_to_jsonl("dataset/job/job_applicant_dataset.csv", "dataset/job/corpus_200_shard0.jsonl")