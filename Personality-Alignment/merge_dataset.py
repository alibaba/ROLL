import json

def merge_datasets():
    """
    合并两个数据集文件：
    - 从dialogue_dataset_all_v2.jsonl读取原始数据
    - 从cleaned_dataset.jsonl读取清理后的输出
    - 将原始数据的output字段替换为清理后的输出
    - 输出到dialogue_dataset_all_v2_cleaned.jsonl
    """
    
    # 读取cleaned_dataset.jsonl中的cleaned_output
    cleaned_outputs = []
    try:
        with open('cleaned_dataset.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                cleaned_outputs.append(data.get('cleaned_output', ''))
        print(f"读取到 {len(cleaned_outputs)} 条清理后的输出")
    except FileNotFoundError:
        print("错误：找不到 cleaned_dataset.jsonl 文件")
        return
    except Exception as e:
        print(f"读取 cleaned_dataset.jsonl 时出错：{e}")
        return
    
    # 读取原始数据集并替换output字段
    processed_count = 0
    try:
        with open('dialogue_dataset_all_v2.jsonl', 'r', encoding='utf-8') as input_file, \
             open('dialogue_dataset_all_v2_cleaned.jsonl', 'w', encoding='utf-8') as output_file:
            
            for line_num, line in enumerate(input_file):
                try:
                    data = json.loads(line.strip())
                    
                    # 检查是否有对应的cleaned_output
                    if line_num < len(cleaned_outputs):
                        data['output'] = cleaned_outputs[line_num]
                        processed_count += 1
                    else:
                        print(f"警告：第 {line_num + 1} 行没有对应的cleaned_output，保持原始output")
                    
                    # 写入处理后的数据
                    output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError as e:
                    print(f"第 {line_num + 1} 行JSON解析错误：{e}")
                    continue
                    
        print(f"处理完成！共处理 {processed_count} 条记录")
        print(f"输出文件：dialogue_dataset_all_v2_cleaned.jsonl")
        
    except FileNotFoundError:
        print("错误：找不到 dialogue_dataset_all_v2.jsonl 文件")
        return
    except Exception as e:
        print(f"处理文件时出错：{e}")
        return

if __name__ == "__main__":
    merge_datasets()