import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from process_data.trajectory_coldstart.common import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a repository-aware train/eval/test split for step-level JSONL data.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--train_output_path", required=True)
    parser.add_argument("--eval_output_path", required=True)
    parser.add_argument("--test_output_path", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--min_eval_repos", type=int, default=2)
    parser.add_argument("--min_test_repos", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_target_kinds", default="action_step,stop_step")
    return parser.parse_args()


def parse_keep_target_kinds(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def summarize_split(rows: List[Dict]) -> Dict:
    repo_counter = Counter(row["repo"] for row in rows)
    target_counter = Counter(row.get("target_kind") for row in rows)
    return {
        "num_samples": len(rows),
        "num_repos": len(repo_counter),
        "target_kinds": dict(target_counter),
    }


def main() -> None:
    args = parse_args()
    keep_target_kinds = parse_keep_target_kinds(args.keep_target_kinds)
    records = [row for row in iter_jsonl(args.input_path) if row.get("target_kind") in keep_target_kinds]

    if not records:
        raise ValueError("No records left after filtering target kinds.")

    by_repo: Dict[str, List[Dict]] = defaultdict(list)
    for row in records:
        by_repo[row["repo"]].append(row)

    repo_items = []
    for repo, rows in by_repo.items():
        target_counter = Counter(row.get("target_kind") for row in rows)
        repo_items.append(
            {
                "repo": repo,
                "rows": rows,
                "num_rows": len(rows),
                "stop_rows": target_counter.get("stop_step", 0),
            }
        )

    rng = random.Random(args.seed)
    stop_repo_items = [item for item in repo_items if item["stop_rows"] > 0]
    non_stop_repo_items = [item for item in repo_items if item["stop_rows"] == 0]
    rng.shuffle(stop_repo_items)
    rng.shuffle(non_stop_repo_items)

    repos_total = len(repo_items)
    train_repo_target = max(1, int(repos_total * args.train_ratio))
    eval_repo_target = max(1, int(repos_total * args.eval_ratio))
    test_repo_target = repos_total - train_repo_target - eval_repo_target

    if repos_total >= 5:
        eval_repo_target = max(eval_repo_target, args.min_eval_repos)
        test_repo_target = max(test_repo_target, args.min_test_repos)
        train_repo_target = repos_total - eval_repo_target - test_repo_target

    if train_repo_target <= 0:
        train_repo_target = 1
        overflow = train_repo_target + eval_repo_target + test_repo_target - repos_total
        if overflow > 0:
            while overflow > 0 and eval_repo_target > 1:
                eval_repo_target -= 1
                overflow -= 1
            while overflow > 0 and test_repo_target > 1:
                test_repo_target -= 1
                overflow -= 1
            train_repo_target = repos_total - eval_repo_target - test_repo_target

    split_targets = {
        "train": train_repo_target,
        "eval": eval_repo_target,
        "test": test_repo_target,
    }
    split_assignments = {"train": [], "eval": [], "test": []}
    split_repo_counts = Counter()
    split_sample_counts = Counter()

    def assign_item(split_name: str, item: Dict) -> None:
        split_assignments[split_name].append(item)
        split_repo_counts[split_name] += 1
        split_sample_counts[split_name] += item["num_rows"]

    # Seed each split with one stop-containing repo when possible.
    for split_name in ("train", "eval", "test"):
        if stop_repo_items and split_repo_counts[split_name] < split_targets[split_name]:
            assign_item(split_name, stop_repo_items.pop())

    remaining_items = stop_repo_items + non_stop_repo_items
    rng.shuffle(remaining_items)

    for item in remaining_items:
        candidate_splits = [
            split_name
            for split_name in ("train", "eval", "test")
            if split_repo_counts[split_name] < split_targets[split_name]
        ]
        if not candidate_splits:
            candidate_splits = ["train", "eval", "test"]

        split_name = min(
            candidate_splits,
            key=lambda name: (
                split_sample_counts[name],
                split_repo_counts[name],
                name,
            ),
        )
        assign_item(split_name, item)

    split_rows = {
        split_name: [row for item in split_assignments[split_name] for row in item["rows"]]
        for split_name in ("train", "eval", "test")
    }

    write_jsonl(args.train_output_path, split_rows["train"])
    write_jsonl(args.eval_output_path, split_rows["eval"])
    write_jsonl(args.test_output_path, split_rows["test"])

    summary = {
        "input_path": args.input_path,
        "keep_target_kinds": sorted(keep_target_kinds),
        "train": summarize_split(split_rows["train"]),
        "eval": summarize_split(split_rows["eval"]),
        "test": summarize_split(split_rows["test"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
