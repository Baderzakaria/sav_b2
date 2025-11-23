from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import glob

import mlflow

try:
    from mlflow.genai.judges import make_judge as mlflow_make_judge
except ImportError:
    mlflow_make_judge = None

PROMPT_FILE = Path("prompts/freemind_prompts.json")
DEFAULT_LOG = Path("data/results/freemind_log_latest.csv")
ISSUE_KEYWORDS = [
    "panne",
    "proble",
    "incident",
    "coupure",
    "rupture",
    "lent",
    "bug",
    "erreur",
    "service",
    "fail",
]

class SimpleJudge:

    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions

    @staticmethod
    def _contains_issue(text: str) -> bool:
        lower = (text or "").lower()
        return any(keyword in lower for keyword in ISSUE_KEYWORDS)

    def __call__(self, inputs: Optional[Dict] = None, outputs: Optional[Dict] = None, **_: Dict) -> Dict[str, str]:
        inputs = inputs or {}
        outputs = outputs or {}
        tweet = inputs.get("tweet", "")
        prediction = str(outputs.get("prediction", "")).lower()
        issue_present = self._contains_issue(tweet)

        if issue_present and prediction not in {"true", "problem", "issue"}:
            verdict = "fail"
            reason = "Tweet looks like an issue but classification flagged otherwise."
        elif not issue_present and prediction in {"true", "problem", "issue"}:
            verdict = "fail"
            reason = "Tweet appears promotional yet classified as an issue."
        elif not prediction or prediction == "unknown":
            verdict = "fail"
            reason = "Agent did not return a usable prediction."
        else:
            verdict = "pass"
            reason = "Classification aligns with heuristic keywords."

        return {"verdict": verdict, "reason": reason}

def make_judge(name: str, instructions: str):
    if mlflow_make_judge is not None:
        return mlflow_make_judge(name=name, instructions=instructions)
    return SimpleJudge(name=name, instructions=instructions)

def _load_judge_config(judge_key: str) -> Dict[str, str]:
    with PROMPT_FILE.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    judges = data.get("judges", {})
    judge_cfg = judges.get(judge_key)
    if not judge_cfg or "instructions" not in judge_cfg:
        raise KeyError(f"Judge '{judge_key}' not found in {PROMPT_FILE}")
    return judge_cfg

def build_judge(judge_key: str = "J_utile_consistency"):
    try:
        cfg = _load_judge_config(judge_key)
        return make_judge(
            name=cfg.get("name", judge_key),
            instructions=cfg["instructions"],
        )
    except (KeyError, FileNotFoundError) as e:

        print(f"Warning: Judge config '{judge_key}' not found, using SimpleJudge fallback: {e}")
        default_instructions = (
            "Evaluate if the 'utile' classification is consistent with the tweet content. "
            "Check if tweets with problems/issues are marked as utile=true, "
            "and promotional/corporate tweets are marked as utile=false."
        )
        return SimpleJudge(name=judge_key, instructions=default_instructions)

def run_judge_on_log(
    log_csv: Path,
    output_csv: Optional[Path] = None,
    judge_key: str = "J_multi_coherence",
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "freemind_judges",
    start_run: bool = True,
    combine_policy: str = "algo_only",
) -> Path:
    log_csv = Path(log_csv)

    if not log_csv.exists():
        candidates = sorted(
            [
                Path(p) for p in glob.glob("data/results/freemind_log_*.csv")
                if "_judge" not in p
            ],
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if candidates:
            log_csv = candidates[0]
            print(f"[judge] Input log not found, using latest by mtime: {log_csv}")
        else:
            raise FileNotFoundError(f"No freemind log files found under data/results/. Expected {DEFAULT_LOG}")
    if output_csv is None:

        if log_csv.name.endswith("_latest.csv"):
            output_csv = log_csv.with_name(log_csv.stem.replace("_latest", "_latest_judge") + ".csv")
        elif "_log" in log_csv.stem:
            output_csv = log_csv.with_name(log_csv.stem.replace("_log", "_judge") + ".csv")
        else:
            output_csv = log_csv.with_name(log_csv.stem + "_judge.csv")
    else:
        output_csv = Path(output_csv)

    judge = build_judge(judge_key)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    def _to_bool_like(val) -> Optional[bool]:
        if val is None:
            return None
        s = str(val).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        return None

    def _to_int(val) -> Optional[int]:
        try:
            return int(float(str(val).strip()))
        except Exception:
            return None

    old_neg = {"colere_extreme", "tres_negatif", "negatif", "mecontent", "legerement_negatif"}
    old_pos = {"legerement_positif", "positif", "tres_positif"}
    old_neu = {"neutre", "mixte"}
    new_neg = {"colere", "frustration", "deception", "inquietude"}
    new_pos = {"satisfaction", "enthousiasme"}
    new_neu = {"neutre"}

    def _sentiment_polarity(sent: str) -> Optional[str]:
        s = (sent or "").strip().lower()
        if s in old_neg or s in new_neg:
            return "neg"
        if s in old_pos or s in new_pos:
            return "pos"
        if s in old_neu or s in new_neu:
            return "neu"
        if not s:
            return None
        return None

    promo_keywords = [
        "dispo", "au prix", "prix", "offre", "promo", "présente", "presente", "la plus", "révolution", "revolution",
        "ultra", "nouvelle box", "smartphone reconditionné", "reconditionné", "reconditionne"
    ]
    pronouns = ["je ", "j'", "ma ", "mes ", "moi ", "mon ", "nous ", "notre "]

    def algorithmic_verdict(row: Dict[str, str]) -> Tuple[str, str]:
        clean_text = (row.get("clean_text") or row.get("full_text") or "").strip().lower()
        final_utile = _to_bool_like(row.get("Final_utile"))
        final_cat = (row.get("Final_categorie") or "").strip().lower()
        final_sent = (row.get("Final_sentiment") or "").strip().lower()
        final_grav = _to_int(row.get("Final_gravity"))
        type_a4 = (row.get("A4_type") or "").strip().lower()
        a1_utile = _to_bool_like(row.get("A1_utile"))

        problem_like = (final_cat in {"probleme", "question"}) or (type_a4 and type_a4 not in {"autre", ""}) or (final_grav is not None and final_grav <= -3)
        if problem_like and final_utile is not True:
            return "fail", "Incoherence: problème/question/type/gravité imply utile=true"

        if final_utile is False and final_grav is not None and not (-1 <= final_grav <= 1):
            return "fail", "Incoherence: utile=false but gravity not in [-1,1]"

        pol = _sentiment_polarity(final_sent)
        if pol == "neg":
            if final_grav is not None and final_grav > -2:
                return "fail", "Incoherence: negative sentiment but gravity too neutral/positive"
        elif pol == "pos":
            if final_grav is not None and final_grav < 2:
                return "fail", "Incoherence: positive sentiment but gravity too neutral/negative"
        elif pol == "neu":
            if final_grav is not None and abs(final_grav) > 1:
                return "fail", "Incoherence: neutral sentiment but gravity not near 0"

        looks_promo = any(k in clean_text for k in promo_keywords) and ("http://" in clean_text or "https://" in clean_text)
        has_personal = any(p in clean_text for p in pronouns)
        if looks_promo and not has_personal and final_utile is True:
            return "fail", "Promo/corporate content marked utile without personal context"

        if a1_utile is not None and final_utile is not None and a1_utile != final_utile and problem_like:
            return "fail", "A1 vs Final utile disagreement on problem-like tweet"

        return "pass", "Classification aligns across agents"

    judged_rows = []
    with log_csv.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:

            algo_verdict, algo_reason = algorithmic_verdict(row)

            llm_verdict = None
            llm_reason = ""
            try:
                inputs = {
                    "clean_text": row.get("clean_text") or row.get("full_text") or "",
                    "A1_utile": row.get("A1_utile"),
                    "A2_categorie": row.get("A2_categorie"),
                    "A3_sentiment": row.get("A3_sentiment"),
                    "A4_type": row.get("A4_type"),
                    "A5_gravity": row.get("A5_gravity"),
                    "Final_utile": row.get("Final_utile"),
                    "Final_categorie": row.get("Final_categorie"),
                    "Final_sentiment": row.get("Final_sentiment"),
                    "Final_gravity": row.get("Final_gravity"),
                }
                outputs = {"prediction": str(row.get("Final_utile")).lower()}
                j = judge(inputs=inputs, outputs=outputs)
                if isinstance(j, dict):
                    llm_verdict = (j.get("verdict") or "").strip().lower() or None
                    llm_reason = (j.get("reason") or "").strip()
                else:
                    llm_verdict = str(j).strip().lower() or None
            except Exception as _:
                llm_verdict = "unavailable"
                llm_reason = "LLM judge unavailable or failed"

            cp = (combine_policy or "algo_only").lower()
            if cp == "llm_only":
                final_agg = llm_verdict if llm_verdict in {"pass", "fail"} else algo_verdict
            elif cp == "either_pass":
                if llm_verdict in {"pass", "fail"}:
                    final_agg = "pass" if (algo_verdict == "pass" or llm_verdict == "pass") else "fail"
                else:
                    final_agg = algo_verdict
            elif cp == "both_strict":
                if llm_verdict in {"pass", "fail"}:
                    final_agg = "pass" if (algo_verdict == "pass" and llm_verdict == "pass") else "fail"
                else:
                    final_agg = algo_verdict
            else:

                final_agg = algo_verdict

            judged_rows.append({
                **row,
                "algo_judge_verdict": algo_verdict,
                "algo_judge_reason": algo_reason,
            })

    def _write_and_log():
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = judged_rows[0].keys() if judged_rows else []
        with output_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(judged_rows)

        try:
            total = len(judged_rows)

            passes = sum(1 for r in judged_rows if str(r.get("algo_judge_verdict", "")).lower() == "pass")
            fails = sum(1 for r in judged_rows if str(r.get("algo_judge_verdict", "")).lower() == "fail")
            algo_pass = passes
            algo_fail = fails
            reason_counts: Dict[str, int] = {}
            for r in judged_rows:
                if str(r.get("algo_judge_verdict", "")).lower() == "fail":
                    reason = str(r.get("algo_judge_reason", "") or "").strip()
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            summary = {
                "log_csv": str(log_csv),
                "output_csv": str(output_csv),
                "total_rows": total,
                "pass": passes,
                "fail": fails,
                "pass_rate": round(passes / total, 4) if total else 0.0,
                "fail_rate": round(fails / total, 4) if total else 0.0,
                "algo": {"pass": algo_pass, "fail": algo_fail},
                "top_fail_reasons": sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                "judge_key": judge_key,
            }
            summary_path = output_csv.with_name(output_csv.stem + "_summary.json")
            with summary_path.open("w", encoding="utf-8") as sfp:
                json.dump(summary, sfp, ensure_ascii=False, indent=2)

            top3 = summary["top_fail_reasons"][:3]
            reasons_str = "; ".join([f"{(reason or 'unclassified')} ({count})" for reason, count in top3])
            print(
                f"[judge] Input: {summary['log_csv']} | Output: {summary['output_csv']}\n"
                f"[judge] Total={total} | Pass={passes} ({summary['pass_rate']*100:.1f}%) | "
                f"Fail={fails} ({summary['fail_rate']*100:.1f}%)\n"
                f"[judge] Algo pass/fail: {algo_pass}/{algo_fail}\n"
                f"[judge] Top fail reasons: {reasons_str}"
            )
        except Exception as _:
            summary_path = None

        mlflow.log_param("judge_log_csv", str(log_csv))
        mlflow.log_param("judge_key", judge_key)
        mlflow.log_artifact(str(output_csv), artifact_path="judge")
        if summary_path:
            mlflow.log_artifact(str(summary_path), artifact_path="judge")

    if start_run or mlflow.active_run() is None:
        with mlflow.start_run(run_name=f"judge_{log_csv.stem}"):
            _write_and_log()
    else:
        _write_and_log()

    return output_csv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLflow judge on pipeline logs.")
    parser.add_argument("--log-csv", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--judge-key", type=str, default="J_multi_coherence")
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns")
    parser.add_argument("--experiment-name", type=str, default="freemind_judges")
    parser.add_argument("--combine-policy", type=str, default="algo_only", choices=["algo_only", "both_strict", "llm_only", "either_pass"])
    return parser.parse_args()

def main():
    args = parse_args()
    result_path = run_judge_on_log(
        log_csv=args.log_csv,
        output_csv=args.output_csv,
        judge_key=args.judge_key,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        start_run=True,
        combine_policy=args.combine_policy,
    )
    print(f"Judge output saved to {result_path}")

if __name__ == "__main__":
    main()

