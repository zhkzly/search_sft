import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_CATEGORIES = ("2_medium_f1_50-100", "3_perfect_match_100")
DEFAULT_LENGTH_BUCKETS = ("short_lt100k", "medium_100k_300k")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]*[a-zA-Z]")


def parse_csv_arg(raw: str | None, default: Sequence[str]) -> List[str]:
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def clean_ansi(text: str) -> str:
    return ANSI_PATTERN.sub("", text)


def clean_tool_result(result: str) -> str:
    text = result or ""
    marker = "\x1b[?2004l"
    if marker in text:
        parts = [part for part in text.split(marker)[1:] if part.strip()]
        if parts:
            text = parts[0]
    return clean_ansi(text).strip()


def json_dumps(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def collect_trajectory_files(
    base_dir: str,
    categories: Sequence[str],
    length_buckets: Sequence[str],
    max_trajectories: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Tuple[Path, str, str]]:
    base_path = Path(base_dir)
    items: List[Tuple[Path, str, str]] = []
    for category in categories:
        for length_bucket in length_buckets:
            target_dir = base_path / category / length_bucket
            if not target_dir.exists():
                continue
            for traj_path in sorted(target_dir.glob("*/trajectory.json")):
                items.append((traj_path, category, length_bucket))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    if max_trajectories is not None:
        return items[:max_trajectories]
    return items


def derive_instance_metadata(base_dir: str, traj_path: Path) -> Dict[str, str]:
    relative = traj_path.relative_to(base_dir)
    quality_bucket = relative.parts[0]
    length_bucket = relative.parts[1]
    instance_id = relative.parts[2]
    repo = instance_id.rsplit("-", 1)[0]
    return {
        "quality_bucket": quality_bucket,
        "length_bucket": length_bucket,
        "instance_id": instance_id,
        "repo": repo,
        "source_path": str(relative),
    }


def iter_jsonl(path: str | Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Sequence[Dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def format_tool_call(raw_tool_call: Dict, fallback_id: str) -> Dict:
    arguments = raw_tool_call.get("arguments", {})
    if isinstance(arguments, str):
        arguments_str = arguments
    else:
        arguments_str = json_dumps(arguments)
    call_id = raw_tool_call.get("call_id") or raw_tool_call.get("id") or fallback_id
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": raw_tool_call.get("name", ""),
            "arguments": arguments_str,
        },
    }

