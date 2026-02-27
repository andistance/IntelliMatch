import csv
import json

def csv_to_jsonl(input_csv_path, output_jsonl_path, resume_column='Resume'):
    """
    读取CSV文件并将每一行的行号和简历保存到JSONL文件中
    
    Args:
        input_csv_path: 输入的CSV文件路径
        output_jsonl_path: 输出的JSONL文件路径
        resume_column: 简历列的列名
    """
    
    records = []
    
    try:
        # 读取CSV文件
        with open(input_csv_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            # 验证简历列是否存在
            if resume_column not in csv_reader.fieldnames:
                raise ValueError(f"CSV文件中不存在名为 '{resume_column}' 的列")
            
            # 遍历每一行
            for row_num, row in enumerate(csv_reader, start=1):  # 从1开始计数
                resume_content = row.get(resume_column, '').strip()
                
                # 构建JSON记录
                record = {
                    "_id": f"job-2026{row_num}",  # 行号作为唯一标识符
                    "text": resume_content
                }
                
                records.append(record)
                
                # 可选：实时进度打印
                if row_num % 100 == 0:
                    print(f"已处理 {row_num} 行...")
                if row_num >= 1000:
                    break
        
        records = records[:100]
        
        # 将记录写入JSONL文件
        with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
            for record in records:
                json_line = json.dumps(record, ensure_ascii=False)
                jsonl_file.write(json_line + '\n')
        
        print(f"转换完成！")
        print(f"处理了 {len(records)} 行数据")
        print(f"结果已保存到: {output_jsonl_path}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_csv_path}'")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_file = "dataset/job/job_applicant_dataset.csv"  # 你的CSV文件路径
    output_file = "dataset/job/queries_100_shard0.jsonl"  # 输出的JSONL文件路径
    resume_column_name = "Resume"  # 简历列的列名
    
    # 执行转换
    csv_to_jsonl(input_file, output_file, resume_column_name)