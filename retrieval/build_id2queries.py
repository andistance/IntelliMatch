import json
import sys
from collections import defaultdict

def merge_jsonl_and_json(jsonl_file_path, json_file_path, output_file_path):
    """
    合并JSONL文件和JSON文件中的数据
    
    Args:
        jsonl_file_path: JSONL文件路径，每个元素有id和text键
        json_file_path: JSON文件路径，每个键是id，值是包含summary和conditions的字典
        output_file_path: 输出JSON文件路径
    """
    
    # 1. 读取JSONL文件
    jsonl_data = {}
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        item = json.loads(line)
                        item_id = item.get('_id')
                        text = item.get('text')
                        
                        if item_id is None or text is None:
                            print(f"警告：JSONL文件第{line_num}行缺少'id'或'text'键")
                            continue
                            
                        jsonl_data[item_id] = {
                            'raw': text
                        }
                    except json.JSONDecodeError as e:
                        print(f"错误：JSONL文件第{line_num}行JSON解析失败: {e}")
                        continue
    except FileNotFoundError:
        print(f"错误：未找到JSONL文件: {jsonl_file_path}")
        return False
    except Exception as e:
        print(f"错误：读取JSONL文件失败: {e}")
        return False
    
    print(f"成功读取JSONL文件，共{len(jsonl_data)}条记录")
    
    # 2. 读取JSON文件
    json_data = {}
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到JSON文件: {json_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"错误：JSON文件解析失败: {e}")
        return False
    except Exception as e:
        print(f"错误：读取JSON文件失败: {e}")
        return False
    
    print(f"成功读取JSON文件，共{len(json_data)}条记录")
    
    # 3. 合并数据
    merged_data = {}
    missing_ids = []
    
    for item_id, json_item in json_data.items():
        if item_id in jsonl_data:
            merged_data[item_id] = {
                'raw': jsonl_data[item_id]['raw'],
                'gpt-4-turbo': {
                    'summary': json_item.get('summary', ''),
                    'conditions': json_item.get('conditions', {})
                }
            }
        else:
            missing_ids.append(item_id)
    
    # 检查是否有JSONL中的ID在JSON中不存在
    jsonl_only_ids = []
    for item_id in jsonl_data:
        if item_id not in json_data:
            jsonl_only_ids.append(item_id)
    
    # 4. 输出统计信息
    print(f"\n数据合并统计：")
    print(f"- 成功合并: {len(merged_data)} 条记录")
    print(f"- JSON中存在但JSONL中不存在: {len(missing_ids)} 条")
    if missing_ids:
        print(f"  缺失的ID示例: {missing_ids[:5]}" + ("..." if len(missing_ids) > 5 else ""))
    print(f"- JSONL中存在但JSON中不存在: {len(jsonl_only_ids)} 条")
    if jsonl_only_ids:
        print(f"  多余的ID示例: {jsonl_only_ids[:5]}" + ("..." if len(jsonl_only_ids) > 5 else ""))
    
    # 5. 保存合并后的数据
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"\n成功保存合并数据到: {output_file_path}")
    except Exception as e:
        print(f"错误：保存输出文件失败: {e}")
        return False
    
    return True

def main():
    # 文件路径配置
    jsonl_file_path = "dataset/job/queries_100_shard0.jsonl"      # JSONL文件路径
    json_file_path = "results/job/retrieval_keywords_shard0_gpt-4-turbo_job.json"        # JSON文件路径
    output_file_path = "dataset/job/id2queries_100_shard0.json"  # 输出文件路径
    
    # 可以在这里修改文件路径
    # 或者从命令行参数获取
    if len(sys.argv) == 4:
        jsonl_file_path = sys.argv[1]
        json_file_path = sys.argv[2]
        output_file_path = sys.argv[3]
    
    print("=" * 50)
    print("JSONL和JSON数据合并工具")
    print("=" * 50)
    print(f"输入文件1 (JSONL): {jsonl_file_path}")
    print(f"输入文件2 (JSON): {json_file_path}")
    print(f"输出文件: {output_file_path}")
    print("=" * 50)
    
    # 执行合并
    success = merge_jsonl_and_json(jsonl_file_path, json_file_path, output_file_path)
    
    if success:
        print("\n✓ 数据合并完成！")
        
        # 显示前几个合并结果的示例
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
            
            if merged_data:
                print("\n合并结果示例（前3个）:")
                for i, (item_id, data) in enumerate(list(merged_data.items())[:3]):
                    print(f"\nID: {item_id}")
                    print(f"  raw (前100字符): {data['raw'][:100]}...")
                    print(f"  gpt-4-turbo.summary: {data['gpt-4-turbo']['summary'][:50]}..." 
                          if data['gpt-4-turbo']['summary'] else "  gpt-4-turbo.summary: (空)")
                    print(f"  gpt-4-turbo.conditions 键数量: {len(data['gpt-4-turbo']['conditions'])}")
        except Exception as e:
            print(f"注意：无法显示结果示例: {e}")
    else:
        print("\n✗ 数据合并失败！")
    
    print("=" * 50)

if __name__ == "__main__":
    main()