"""保护既有原始记录，并为新增单卡与E5逐步版本建立独立哈希清单。"""
from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify_manifest(path, prefix=ROOT):
    entries = json.loads(path.read_text())
    for name, expected in entries.items():
        if digest(prefix / name) != expected:
            raise ValueError(f"已有原始文件哈希变化：{prefix / name}")
    return {"manifest": str(path.relative_to(ROOT)), "files_checked": len(entries), "all_unchanged": True}


def make_manifest(paths, target, expected_count):
    if len(paths) != expected_count:
        raise ValueError(f"原始文件数错误：{target}: {len(paths)} != {expected_count}")
    values = {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}
    if target.exists() and json.loads(target.read_text()) != values:
        raise ValueError(f"已存在的原始保护表与当前文件不一致：{target}")
    target.write_text(json.dumps(values, indent=2) + "\n")
    return {"manifest": str(target.relative_to(ROOT)), "files_hashed": len(values)}


def run():
    old = [verify_manifest(ROOT / "archive/E0_before_seed_report_20261116_194619/original_gpu_files_sha256.json")]
    for experiment in ("E1", "E2", "E3", "E4"):
        old.append(verify_manifest(ROOT / experiment / "results/original_gpu_files_sha256.json"))
    previous = json.loads((ROOT / "E5/results/previous_round_only_results.json").read_text())
    archived = ROOT / previous["archive"]
    old.append(verify_manifest(archived / "E5/results/original_gpu_files_sha256.json", archived))

    additions = []
    for experiment, count in (("E2", 450), ("E3", 360), ("E4", 450), ("E5", 585)):
        module = importlib.import_module(f"{experiment}.single_gpu.config")
        paths = []
        for case in module.cases_for():
            folder = case.folder()
            files = [path for path in folder.iterdir() if path.is_file()]
            expected_files = 11 if case.role == "serial_local_sgd" else 10
            if len(files) != expected_files:
                raise ValueError(f"运行文件数错误：{folder}")
            paths.extend(files)
        target = module.BASELINE_ROOT / "original_gpu_files_sha256.json"
        additions.append(make_manifest(paths, target, count))
        snapshot = module.BASELINE_ROOT / "provenance/source_sha256.json"
        entries = json.loads(snapshot.read_text())
        for name, expected in entries.items():
            if name.endswith(("check_results.py", "compare.py", "plot_results.py")):
                continue
            if digest(ROOT / name) != expected:
                raise ValueError(f"新增单卡训练来源变化：{name}")

    paths = []
    for dataset in ("mnist", "fashionmnist", "cifar10"):
        for h in (1, 2, 4, 8, 16):
            for seed in (1234, 1235, 1236):
                folder = ROOT / f"E5/results/{dataset}/h{h}/seed{seed}"
                files = [path for path in folder.iterdir() if path.is_file()]
                if len(files) != 16:
                    raise ValueError(f"E5逐步版本文件数错误：{folder}")
                paths.extend(files)
    additions.append(make_manifest(paths, ROOT / "E5/results/original_gpu_files_sha256.json", 720))
    result = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "previous_raw_files": old, "new_raw_files": additions,
              "single_gpu_training_sources_unchanged": True, "all_checks_passed": True}
    (ROOT / "docs/原始记录完整性核验.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Original E0-E4 and archived E5 hashes unchanged; new raw-file manifests recorded", flush=True)
    return result


if __name__ == "__main__":
    run()
