import json

# 加载转换后的数据
with open("sft/data/training_data.json") as f:
    data = json.load(f)

# 检查第一条数据
sample = data[0]
print("Input length:", len(sample["input"]))
print("\nOutput items:", len(sample["output"]))

# 统计loss计算的token数
gen_count = sum(1 for item in sample["output"] if "gen" in item)
doc_gen_count = sum(1 for item in sample["output"] if "doc_gen" in item)

print(f"\nItems with loss: {gen_count}")
print(f"Items without loss: {doc_gen_count}")

# 显示前3个output
for i, item in enumerate(sample["output"][:3]):
    print(f"\n--- Output {i+1} ---")
    for k, v in item.items():
        print(f"{k}: {v[:200]}..." if len(str(v)) > 200 else f"{k}: {v}")
