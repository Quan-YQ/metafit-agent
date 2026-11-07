"""交付自检：公开研究链和模拟候选，不联网、不读取受限数据内容。"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_vision_choice import OfflineNoObservation, run_pair
from fusionposer_cross_record import evaluate, leave_one_record_out, load_recording
from fusionposer_fetch import check_or_fetch
from research_motion_choice import OfflineChoiceModel
from trace_motion_agent import ScriptedObserver


ROOT = Path(__file__).resolve().parents[1]


def run():
    files = check_or_fetch()
    if any(item["status"] != "verified_existing" for item in files):
        raise ValueError("缺少公开数据；执行 fusionposer_fetch.py --download")
    records = [(name, load_recording(name)) for name in ("train", "test_1", "test_2")]
    cross = evaluate(records[0][1], records[1:])
    cross["leave_one_record_out"] = leave_one_record_out(records)
    event = json.loads((ROOT / "examples/set_completed.json").read_text(encoding="utf-8"))
    event["exercise_id"] = "research_motion"
    for field in ("planned_reps", "completed_reps", "weight_kg_per_hand", "metrics", "set_index", "rest_seconds"):
        event.pop(field, None)
    event["discomfort"] = {"status": "none", "source": "self_report"}
    pairs = []
    for sample, answer in (("round_a", "up"), ("squat_1", "down")):
        result = run_pair(event, sample, ScriptedObserver(answer), OfflineChoiceModel(), OfflineNoObservation())
        if (result["visual_result"]["r3_status"] != "reviewed" or result["text_only"]["review"] != "reviewed"
                or result["delivery"] or result["may_display"]):
            raise ValueError("离线候选链验证失败")
        pairs.append({"sample": sample, "action_comparison": result["action_comparison"]})
    return {"status": "research_demo_ready", "network": False, "delivery": False,
            "public_data_files_verified": len(files), "offline_candidate_checks": pairs,
            "cross_record_evaluation": cross,
            "licensed_archives_present": {"amass": (ROOT / "data/amass/ACCAD.tar.bz2").is_file(),
                                          "body_model": (ROOT / "data/body_models/smplh.tar.xz").is_file()},
            "notice": "仅评审演示可运行；离线选择为替身，非模型能力或真实训练安全验收。"}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, IndexError):
        print("自检未完成；请核对公开样本后再运行。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
