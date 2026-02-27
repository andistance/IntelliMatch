import json
import sys
from collections import defaultdict

def process_candidate_jobs(candidates_jsonl_file, candidate_jobs_file, jobs_jsonl_file, output_file):
    """
    处理应聘者职位数据
    
    Args:
        candidate_jobs_file: JSON文件路径，包含应聘者ID到职位ID列表的映射
        jobs_jsonl_file: JSONL文件路径，包含每个职位的详细信息
        output_file: 输出JSON文件路径
    """
    
    # 1. 读取应聘者职位映射文件
    with open(candidate_jobs_file, 'r', encoding='utf-8') as f:
        candidate_jobs_mapping = json.load(f)
    
    # 2. 构建职位ID到职位详情的索引
    job_index = {}
    
    with open(jobs_jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    job_data = json.loads(line)
                    # 假设职位字典中有'job_id'键
                    job_id = job_data.get('_id')
                    cleaned_data = {k: v for k, v in job_data.get('metadata', {}).items()}
                    cleaned_data['NCTID'] = job_id  # 将职位ID添加到清洗后的数据中
                    if job_id is not None:
                        job_index[job_id] = cleaned_data
                except json.JSONDecodeError as e:
                    print(f"警告：跳过无效的JSONL行: {e}")
    
    candidates_index = {}

    with open(candidates_jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidate_data = json.loads(line)
                    candidate_id = candidate_data.get('_id')
                    if candidate_id is not None:
                        candidates_index[candidate_id] = candidate_data
                except:
                    print(f"警告：跳过无效的应聘者JSONL行: {line}")
    
    # 3. 处理每个应聘者的数据
    result = []
    
    for candidate_id, job_ids in candidate_jobs_mapping.items():
        candidate_result = {
            "candidate_id": candidate_id,
            "candidate": candidates_index.get(candidate_id, {}).get('text', ""),
            "0": [],  # 前1-5个职位
            "1": [],  # 第6-15个职位
            "2": []   # 第16-30个职位
        }
        
        # 获取前30个职位ID（如果不足30个，则获取所有）
        job_ids_to_process = job_ids[:30]
        
        # 分段处理职位ID
        for i, job_id in enumerate(job_ids_to_process, start=1):  # 从1开始计数
            if job_id in job_index:
                job_data = job_index[job_id]
                
                if 1 <= i <= 5:
                    candidate_result["0"].append(job_data)
                elif 6 <= i <= 15:
                    candidate_result["1"].append(job_data)
                elif 16 <= i <= 30:
                    candidate_result["2"].append(job_data)
            else:
                print(f"警告：应聘者 {candidate_id} 的职位ID {job_id} 在JSONL文件中未找到")
        
        result.append(candidate_result)
    
    # 4. 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理了 {len(result)} 个应聘者的数据")
    print(f"结果已保存到: {output_file}")
    
    return result

def main():
    # 文件路径配置
    candidates_jsonl_file = "dataset/job/queries_100_shard0.jsonl" # 应聘者JSON文件路径
    candidate_jobs_file = "results/job/qid2nctids_results_gpt-4-turbo_job_k20_bm25wt1_medcptwt1_N2000.json"  # 应聘者职位映射JSON文件
    jobs_jsonl_file = "dataset/job/corpus_200_shard0.jsonl"           # 职位详情JSONL文件
    output_file = "dataset/job/retrieved_jobs_100_shard0.json"  # 输出文件
    
    try:
        # 处理数据
        result = process_candidate_jobs(
            candidates_jsonl_file,
            candidate_jobs_file, 
            jobs_jsonl_file, 
            output_file
        )
        
        # 显示处理结果的统计信息
        print("\n统计信息：")
        print(f"总共处理了 {len(result)} 个应聘者")
        
        #for i in range(min(3, len(result))):  # 显示前3个应聘者的简要信息
        #    candidate_data = result[i]
        #    print(f"应聘者 {candidate_data['candidate_id']}:")
        #    print(f"  列表0: {len(candidate_data['0'])} 个职位")
        #    print(f"  列表1: {len(candidate_data['1'])} 个职位")
        #    print(f"  列表2: {len(candidate_data['2'])} 个职位")
            
    except FileNotFoundError as e:
        print(f"错误：文件未找到 - {e}")
        print("请确保以下文件存在：")
        print(f"1. {candidate_jobs_file}")
        print(f"2. {jobs_jsonl_file}")
    except json.JSONDecodeError as e:
        print(f"错误：JSON解析失败 - {e}")
    except Exception as e:
        print(f"错误：处理过程中发生异常 - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()