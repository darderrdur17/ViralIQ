"""Generate evaluation, executive and model-card documents from a training run.

Every number written here is read from ``run.json``. If a figure was not measured,
the report says "not measured" rather than filling the gap.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.config import SETTINGS
from src.common.logging import get_logger

log = get_logger("report")


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _pct(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _lift(value) -> str:
    return "—" if value is None else f"{value:,.2f}×"


def _ranking_key(entry: dict) -> str | None:
    ranking = (entry.get("test") or {}).get("ranking") or {}
    return next(iter(ranking), None)


def render_benchmark(run: dict) -> str:
    metrics = run["metrics"]
    champion = _champion(run)
    key = _ranking_key(metrics.get(champion, {})) if champion else None
    k_label = key.split("=")[1] if key else "—"

    lines = [
        "# Model Benchmark",
        "",
        f"Run: `{run['run_id']}`  ",
        f"Trained: {run['started_at']}  ",
        f"Wall time: {run['duration_seconds']} s  ",
        f"Samples: {', '.join(run['sample_files'])}",
        "",
        "## Splits",
        "",
        "| Split | Rows | From | To |",
        "|---|---|---|---|",
    ]
    for name, info in (run.get("splits") or {}).items():
        lines.append(
            f"| {name} | {_fmt(info['rows'])} | {str(info['min_posted_at'])[:19]} | "
            f"{str(info['max_posted_at'])[:19]} |"
        )

    recent_rows_count = ((run.get("splits") or {}).get("recent") or {}).get("rows") or 0
    if recent_rows_count == 0:
        lines += [
            "",
            "> No separate forward holdout was available for this run (only two months were "
            "ingested). The `test` split is still strictly later in time than `train`, so the "
            "reported numbers are out-of-time rather than in-sample, but they rest on a single "
            "month boundary. Treat drift conclusions as provisional.",
        ]

    lines += [
        "",
        "## Headline comparison (test split)",
        "",
        f"Ranking metrics at K = {k_label}. Prevalence shown so Precision@K can be read "
        "against the base rate.",
        "",
        f"| Model | PR-AUC | PR-AUC 95% CI | ROC-AUC | Precision@{k_label} | Lift@{k_label} | "
        "Brier | ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(
        metrics.items(),
        key=lambda kv: -((kv[1].get("test") or {}).get("pr_auc") or -1),
    )
    for name, entry in ordered:
        test = entry.get("test") or {}
        rank = (test.get("ranking") or {}).get(key, {}) if key else {}
        ci = entry.get("test_pr_auc_ci") or {}
        ci_txt = (
            f"{_fmt(ci.get('lower'), 3)}–{_fmt(ci.get('upper'), 3)}"
            if ci.get("lower") is not None
            else "—"
        )
        marker = " **(champion)**" if name == champion else ""
        if entry.get("error"):
            lines.append(f"| {name}{marker} | failed | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name}{marker} | {_fmt(test.get('pr_auc'))} | {ci_txt} | "
            f"{_fmt(test.get('roc_auc'))} | {_pct(rank.get('precision_at_k'))} | "
            f"{_lift(rank.get('lift_at_k'))} | {_fmt(test.get('brier'))} | "
            f"{_fmt(test.get('ece_10'))} |"
        )
    lines += ["", f"Test-set prevalence: {_pct((metrics.get(champion, {}).get('test') or {}).get('prevalence'), 3)}", ""]

    # ---- computed interpretation (no hand-written claims) -----------------
    champ_test_for_reading = metrics.get(champion, {}).get("test") or {}
    prev = champ_test_for_reading.get("prevalence")
    pr = champ_test_for_reading.get("pr_auc")
    rank_for_reading = (champ_test_for_reading.get("ranking") or {}).get(key, {}) if key else {}
    if prev and pr:
        ratio = pr / prev
        lines += [
            "### Reading these numbers",
            "",
            f"- PR-AUC is **{ratio:.2f}×** the base rate. A random ranker scores {prev:.4f}; "
            f"the champion scores {pr:.4f}. This is real but modest discrimination — not a "
            "strong predictor.",
            f"- Precision@{k_label} lift is **{_lift(rank_for_reading.get('lift_at_k'))}** over "
            "random selection at the same cutoff.",
        ]
        cal_rows = champ_test_for_reading.get("calibration") or []
        if cal_rows:
            total = sum(r["count"] for r in cal_rows) or 1
            mean_pred = sum(r["mean_predicted"] * r["count"] for r in cal_rows) / total
            mean_obs = sum(r["observed_rate"] * r["count"] for r in cal_rows) / total
            if mean_obs > 0:
                direction = "over-predicts" if mean_pred > mean_obs else "under-predicts"
                lines.append(
                    f"- The model {direction} on average: mean predicted probability "
                    f"{mean_pred:.5f} against an observed rate of {mean_obs:.5f} "
                    f"(ratio {mean_pred / mean_obs:.2f}×). Probabilities should be read as a "
                    "ranking signal, not as a literal risk percentage."
                )
        lines.append("")

    # Recent holdout
    recent_rows = [
        (n, e) for n, e in metrics.items() if (e.get("recent") or {}).get("pr_auc") is not None
    ]
    if recent_rows:
        lines += [
            "## Forward-time holdout (untouched later month)",
            "",
            f"| Model | Rows | Prevalence | PR-AUC | Precision@{k_label} | Lift@{k_label} |",
            "|---|---|---|---|---|---|",
        ]
        for name, entry in sorted(recent_rows, key=lambda kv: -(kv[1]["recent"]["pr_auc"] or 0)):
            r = entry["recent"]
            rank = (r.get("ranking") or {}).get(key, {}) if key else {}
            lines.append(
                f"| {name} | {_fmt(r.get('n'))} | {_pct(r.get('prevalence'), 3)} | "
                f"{_fmt(r.get('pr_auc'))} | {_pct(rank.get('precision_at_k'))} | "
                f"{_lift(rank.get('lift_at_k'))} |"
            )
        lines.append("")

    lines += ["## Calibration (champion, test split)", ""]
    champ_test = metrics.get(champion, {}).get("test") or {}
    cal = champ_test.get("calibration") or []
    if cal:
        lines += ["| Predicted band | Count | Mean predicted | Observed rate |", "|---|---|---|---|"]
        for row in cal:
            lines.append(
                f"| {row['bin_lower']:.5f}–{row['bin_upper']:.5f} | {_fmt(row['count'])} | "
                f"{row['mean_predicted']:.5f} | {row['observed_rate']:.5f} |"
            )
    else:
        lines.append("Calibration table not available for this run.")
    lines.append("")

    imp = (run.get("feature_importance") or {}).get(champion) or []
    if imp:
        lines += ["## Feature importance (champion)", "", "| Feature | Importance |", "|---|---|"]
        for row in imp[:15]:
            lines.append(f"| {row['feature']} | {row['importance']:.5f} |")
        lines.append("")

    lines += [
        "## How to read this",
        "",
        "- **PR-AUC** is the primary metric. A random ranker scores approximately the prevalence, "
        "so compare PR-AUC against the prevalence line above, not against 0.5.",
        "- **Lift@K** states how many times better than random the top-K selection is.",
        "- **Brier** and **ECE** measure probability quality. A model can rank well and still be "
        "badly calibrated.",
        "- Confidence intervals resample whole sounds, because videos sharing a sound are "
        "correlated.",
        "",
        "## Limitations",
        "",
        "See `docs/limitations.md`. In short: this is a retrospective cohort-relative snapshot "
        "model. It is not a seven-day view forecaster, and no fixed-age label was available.",
        "",
    ]
    return "\n".join(lines)


def render_executive(run: dict) -> str:
    metrics = run["metrics"]
    champion = _champion(run)
    test = (metrics.get(champion) or {}).get("test") or {}
    diag = (run.get("label_diagnostics") or {}).get("train") or {}
    key = _ranking_key(metrics.get(champion, {}))
    rank = (test.get("ranking") or {}).get(key, {}) if key else {}
    ci = (metrics.get(champion) or {}).get("test_pr_auc_ci") or {}

    baseline = (metrics.get("E0_random_ranking") or {}).get("test") or {}
    base_rank = (baseline.get("ranking") or {}).get(key, {}) if key else {}

    # Best gradient-boosted competitor, so the report can state whether the extra
    # model capacity actually paid for itself.
    boosted = {
        name: (entry.get("test") or {}).get("pr_auc")
        for name, entry in metrics.items()
        if name.startswith(("E3", "E4")) and (entry.get("test") or {}).get("pr_auc") is not None
    }
    best_boosted_name = max(boosted, key=boosted.get) if boosted else None
    best_boosted = boosted.get(best_boosted_name) if best_boosted_name else None
    champ_is_boosted = champion is not None and champion.startswith(("E3", "E4"))
    linear_pr = ((metrics.get("E1_logistic_regression") or {}).get("test") or {}).get("pr_auc")

    if best_boosted and not champ_is_boosted:
        model_note = (
            f"- The champion is a **linear** model. The strongest gradient-boosted candidate "
            f"({best_boosted_name}) reached {_fmt(best_boosted)}, so the extra capacity did not "
            "pay for itself on this feature set. That is a finding, not an oversight."
        )
    elif best_boosted:
        model_note = (
            f"- A gradient-boosted model won. The best linear baseline reached {_fmt(linear_pr)}."
        )
    else:
        model_note = "- No gradient-boosted model completed in this run."

    return "\n".join(
        [
            "# Executive Summary",
            "",
            f"**Run** `{run['run_id']}` · {run['started_at']}",
            "",
            "## What was built",
            "",
            "A TikTok virality intelligence platform that scores videos from pre-publication "
            "metadata only, explains each score, and reports its own data freshness and model "
            "health. The inference path is a live API, not a notebook.",
            "",
            "## What the model does",
            "",
            f"- Champion model: **{champion}**",
            f"- Test PR-AUC: **{_fmt(test.get('pr_auc'))}** "
            f"(95% CI {_fmt(ci.get('lower'), 3)}–{_fmt(ci.get('upper'), 3)})",
            f"- Random-ranking PR-AUC for reference: {_fmt(baseline.get('pr_auc'))}",
            f"- Precision@K: **{_pct(rank.get('precision_at_k'))}** vs "
            f"{_pct(base_rank.get('precision_at_k'))} for random selection "
            f"(**{_lift(rank.get('lift_at_k'))} lift**)",
            f"- Base rate: {_pct(test.get('prevalence'), 3)}",
            model_note,
            "",
            "## Key caveat that shapes everything",
            "",
            f"{diag.get('pct_excluded_invalid_clock', '—')}% of training rows were excluded "
            "because their observation clock was invalid: engagement statistics timestamped "
            "before publication (physically impossible), or older than the configured horizon.",
            "",
            "Because of that, the shipped label is a cohort-relative snapshot label. It answers "
            "\"does this video land in its peer group's top 1% of observed views\", not \"how many "
            "views will it have in seven days\". Fixed-age labels are emitted as NULL rather than "
            "estimated.",
            "",
            "## Recommendation",
            "",
            "Treat the current model as a validated **ranking prior** for content triage, and "
            "invest in an observation pipeline before making forecasting claims. The highest-value "
            "next step is not a better model — it is repeated measurements of the same videos.",
            "",
        ]
    )


def render_model_card(run: dict) -> str:
    metrics = run["metrics"]
    champion = _champion(run)
    entry = metrics.get(champion) or {}
    test = entry.get("test") or {}
    th = run.get("thresholds") or {}
    diag = (run.get("label_diagnostics") or {}).get("train") or {}
    env = run.get("environment") or {}

    return "\n".join(
        [
            "# Model Card",
            "",
            f"**Version** `{champion}@{run['run_id']}`",
            "",
            "## Intended use",
            "",
            "Ranking candidate videos by their probability of landing in the top 1% of observed "
            "views within their country-and-format peer group. Intended for content triage and "
            "analyst prioritisation, with human review.",
            "",
            "## Out-of-scope use",
            "",
            "- Predicting exact view counts or seven-day outcomes.",
            "- Any causal claim about why a video performed well.",
            "- Automated publishing or removal decisions.",
            "- Comparing creators, since follower counts are absent from the source.",
            "",
            "## Training data",
            "",
            "- Source: `datasocial/tiktok-5.6B-videos` (CC BY-NC 4.0)",
            f"- Samples: {', '.join(run['sample_files'])}",
            f"- Training rows: {_fmt(diag.get('rows'))}",
            f"- Rows excluded for an invalid observation clock: "
            f"{diag.get('pct_excluded_invalid_clock', '—')}%",
            "",
            "## Label",
            "",
            f"- Definition: top {100 * (1 - th.get('percentile', 0.99)):.0f}% of observed views "
            f"within `{' × '.join(th.get('keys', []))}` cohorts",
            f"- Global threshold: {_fmt(round(th.get('global_threshold', 0)))} views",
            f"- Cohort thresholds fitted: {_fmt(th.get('cohort_count'))}",
            f"- Prevalence on valid rows: {_pct(diag.get('prevalence_valid'), 3)}",
            "- Fixed-age labels (`viral_24h`, `viral_7d`): **unavailable by design**",
            "",
            "## Performance",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| PR-AUC | {_fmt(test.get('pr_auc'))} |",
            f"| ROC-AUC | {_fmt(test.get('roc_auc'))} |",
            f"| Brier | {_fmt(test.get('brier'))} |",
            f"| ECE (10 bins) | {_fmt(test.get('ece_10'))} |",
            f"| Test rows | {_fmt(test.get('n'))} |",
            "",
            "## Ethical and legal considerations",
            "",
            "- The source dataset is **non-commercial only**. Commercial deployment requires a "
            "separate licence.",
            "- No creator identifiers or follower data are used.",
            "- Predictions are statistical associations, not judgements about creators.",
            "",
            "## Environment",
            "",
            f"- Python {env.get('python', '—')}",
            f"- Platform {env.get('platform', '—')}",
            "",
        ]
    )


def _champion(run: dict) -> str | None:
    # Reports describe the model that is serving, which is not always the PR-AUC winner:
    # a demo can pin a champion, and a pinned run would otherwise publish a model card
    # contradicting the registry.
    override = run.get("champion_override")
    if override:
        return override
    scored = []
    for name, entry in run["metrics"].items():
        if name.startswith("E0"):
            continue  # baselines and heuristics are never the champion
        pr = (entry.get("test") or {}).get("pr_auc")
        if pr is not None:
            scored.append((pr, name))
    return max(scored)[1] if scored else None


def write_reports(run_path: Path, champion: str | None = None) -> dict[str, str]:
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    if champion:
        run["champion_override"] = champion
    outputs = {
        "benchmark": SETTINGS.paths.reports_evaluation / "model_benchmark.md",
        "executive": SETTINGS.paths.reports_executive / "executive_summary.md",
        "model_card": SETTINGS.paths.root / "docs" / "model_card.md",
    }
    renderers = {
        "benchmark": render_benchmark,
        "executive": render_executive,
        "model_card": render_model_card,
    }
    written = {}
    for key, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(renderers[key](run), encoding="utf-8")
        written[key] = str(path)
        log.info("wrote %s", path)
    return written
