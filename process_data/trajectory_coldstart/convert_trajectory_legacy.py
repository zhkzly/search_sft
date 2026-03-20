import json
import os
import re
from typing import List, Dict
from pathlib import Path

def clean_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]*[a-zA-Z]', '', text)

def format_tool_call(tool_call: Dict) -> str:
    name = tool_call.get("name", "")
    args = json.dumps(tool_call.get("arguments", {}))
    return f"<tool_call>{name}</tool_call><args>{args}</args>"

def format_tool_result(tool_result: Dict) -> str:
    result = tool_result.get("result", "")
    # 用 \x1b[?2004l 分割，取第二部分（跳过命令回显）
    marker = '\x1b[?2004l'
    if marker in result:
        parts = result.split(marker)
        # parts[0] 是命令回显，parts[1] 是实际输出，parts[-1] 可能是空的结尾
        result = parts[1] if len(parts) > 1 else result
    # 清洗剩余的 ANSI 控制符
    result = clean_ansi(result).strip()
    return f"<tool_result>{result}</tool_result>"

def convert_single_trajectory(traj_data: Dict) -> Dict:
    task = traj_data.get("task", "")
    interactions = traj_data.get("llm_interactions", [])
    
    if not interactions:
        return None
    
    # 从第一个interaction获取system prompt
    first_interaction = interactions[0]
    system_prompt = ""
    for msg in first_interaction.get("input_messages", []):
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
            break
    
    input_text = f"{task}\n\n{system_prompt}" if system_prompt else task
    
    output = []
    for interaction in interactions:
        # 1. 先处理 input_messages 中的 tool_result (来自上一轮的工具执行结果)
        for msg in interaction.get("input_messages", []):
            if msg.get("role") == "user" and msg.get("tool_result"):
                tool_result = msg["tool_result"]
                output.append({"doc_gen": format_tool_result(tool_result)})
        
        # 2. 处理 response
        response = interaction.get("response", {})
        
        # 2a. 推理内容 - 计算loss
        content = response.get("content", "")
        if content and content.strip():
            output.append({"gen": content})
        
        # 2b. 工具调用 - 计算loss
        tool_calls = response.get("tool_calls") or []
        for tool_call in tool_calls:
            output.append({"gen": format_tool_call(tool_call)})
    
    return {"input": input_text, "output": output}

def scan_trajectories(base_dir: str, categories: List[str], length_types: List[str]) -> List[str]:
    trajectory_files = []
    base_path = Path(base_dir)
    
    for category in categories:
        for length_type in length_types:
            target_dir = base_path / category / length_type
            if target_dir.exists():
                for task_dir in target_dir.iterdir():
                    if task_dir.is_dir():
                        traj_file = task_dir / "trajectory.json"
                        if traj_file.exists():
                            trajectory_files.append(str(traj_file))
    return trajectory_files

def main():
    base_dir = "classified_by_f1"
    output_file = "sft/data/trajectory_training_data.json"
    categories = ["2_medium_f1_50-100", "3_perfect_match_100"]
    length_types = ["short_lt100k"]
    
    print(f"Scanning from {base_dir}...")
    trajectory_files = scan_trajectories(base_dir, categories, length_types)
    print(f"Found {len(trajectory_files)} trajectory files")
    
    converted = []
    for i, traj_file in enumerate(trajectory_files):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                traj_data = json.load(f)
            result = convert_single_trajectory(traj_data)
            if result:
                converted.append(result)
        except Exception as e:
            print(f"Error: {traj_file}: {e}")
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(trajectory_files)}")
    
    print(f"\nConverted {len(converted)}/{len(trajectory_files)} trajectories")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
