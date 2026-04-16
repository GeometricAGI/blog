"""Validate all figures, numbers, and results in the torch.compile mode study blog post.

This test reads the raw benchmark CSV and recomputes every claim made in
scripts/research/blog.md, ensuring the blog stays in sync with the data.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

CSV_PATH = Path("raw_results.csv")
BLOG_PATH = Path("blog.md")

ALL_MODES = [
    "eager",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]

# Workloads excluded from decode analysis because their input shapes are
# invariant to the shape preset (training-only losses or fixed spatial dims).
DECODE_EXCLUDED = [
    "vlm_contrastive_nce_loss",
    "diff_vae_decoder_upsample",
    "diff_time_embedding_mlp",
    "rl_dpo_loss",
    "rl_grpo_loss",
]


def _load_and_dedup(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["status"] == "ok"].drop_duplicates(
        subset=["shape_preset", "workload", "mode"]
    )


def _filter_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    sub = df[df["shape_preset"] == preset]
    if preset == "decode":
        sub = sub[~sub["workload"].isin(DECODE_EXCLUDED)]
    return sub


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Load and deduplicate the raw results CSV."""
    assert CSV_PATH.exists(), f"Missing raw results file: {CSV_PATH}"
    return _load_and_dedup(CSV_PATH)


@pytest.fixture(scope="module")
def blog_text() -> str:
    """Load the blog markdown."""
    assert BLOG_PATH.exists(), f"Missing blog file: {BLOG_PATH}"
    return BLOG_PATH.read_text()


@pytest.fixture(scope="module")
def wins_by_preset(raw_df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Compute winning mode and workload lists per preset."""
    result: dict[str, dict[str, list[str]]] = {}
    for preset in ["prefill", "decode"]:
        sub = _filter_preset(raw_df, preset)
        idx = sub.groupby("workload")["median_ms"].idxmin()
        winners = sub.loc[idx][["workload", "mode"]]
        mode_to_workloads: dict[str, list[str]] = {}
        for _, row in winners.iterrows():
            mode_to_workloads.setdefault(row["mode"], []).append(row["workload"])
        for mode in mode_to_workloads:
            mode_to_workloads[mode] = sorted(mode_to_workloads[mode])
        result[preset] = mode_to_workloads
    return result


@pytest.fixture(scope="module")
def gmean_by_preset(raw_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute geometric mean speedup vs eager per (preset, mode)."""
    result: dict[str, dict[str, float]] = {}
    for preset in ["prefill", "decode"]:
        sub = _filter_preset(raw_df, preset)
        base = sub[sub["mode"] == "eager"][["workload", "median_ms"]].rename(
            columns={"median_ms": "eager_ms"}
        )
        merged = sub.merge(base, on="workload", how="inner")
        merged["speedup"] = merged["eager_ms"] / merged["median_ms"]
        gmean = (
            merged.groupby("mode")["speedup"]
            .apply(lambda s: math.exp(s.clip(lower=1e-9).map(math.log).mean()))
            .to_dict()
        )
        result[preset] = gmean
    return result


@pytest.fixture(scope="module")
def win_margins_by_preset(raw_df: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """Compute head-to-head win margins between default and mancg."""
    result: dict[str, dict[str, dict]] = {}
    for preset in ["prefill", "decode"]:
        sub = _filter_preset(raw_df, preset)
        pivot = sub.pivot_table(
            index="workload", columns="mode", values="median_ms"
        ).reset_index()
        pivot["margin_pct"] = (
            (pivot["default"] - pivot["max-autotune-no-cudagraphs"])
            / pivot["default"]
            * 100
        )
        pivot["h2h_winner"] = pivot["margin_pct"].apply(
            lambda x: "max-autotune-no-cudagraphs" if x > 0 else "default"
        )
        preset_result = {}
        for mode in ["default", "max-autotune-no-cudagraphs"]:
            mode_wins = pivot[pivot["h2h_winner"] == mode]
            margins = mode_wins["margin_pct"].abs()
            preset_result[mode] = {
                "wins": len(mode_wins),
                "avg_margin": float(margins.mean()) if len(margins) > 0 else 0.0,
            }
        result[preset] = preset_result
    return result


@pytest.fixture(scope="module")
def domain_wins_by_preset(raw_df: pd.DataFrame) -> dict[str, dict]:
    """Compute wins by domain for default vs mancg."""
    result: dict[str, dict] = {}
    two_modes = ["default", "max-autotune-no-cudagraphs"]
    for preset in ["prefill", "decode"]:
        sub = _filter_preset(raw_df, preset)
        idx = sub.groupby("workload")["median_ms"].idxmin()
        winners = sub.loc[idx][["workload", "domain", "mode"]]
        w2 = winners[winners["mode"].isin(two_modes)]
        counts = (
            w2.groupby(["domain", "mode"]).size().unstack(fill_value=0).to_dict()
        )
        result[preset] = counts
    return result


# ── Total workload count ─────────────────────────────────────────────


def test_total_workload_instances(raw_df: pd.DataFrame, blog_text: str):
    """Blog claims 41 workload instances (23 prefill + 18 decode)."""
    assert "41 workload instances" in blog_text
    n_prefill = raw_df[raw_df["shape_preset"] == "prefill"]["workload"].nunique()
    decode_df = _filter_preset(raw_df, "decode")
    n_decode = decode_df["workload"].nunique()
    assert n_prefill + n_decode == 41


def test_prefill_workload_count(raw_df: pd.DataFrame):
    """Prefill should have 23 unique workloads."""
    n = raw_df[raw_df["shape_preset"] == "prefill"]["workload"].nunique()
    assert n == 23, f"prefill has {n} workloads, expected 23"


def test_decode_workload_count(raw_df: pd.DataFrame):
    """Decode should have 18 unique workloads after excluding invariant ones."""
    decode_df = _filter_preset(raw_df, "decode")
    n = decode_df["workload"].nunique()
    assert n == 18, f"decode has {n} workloads, expected 18"


# ── Prefill wins ─────────────────────────────────────────────────────


def test_prefill_win_counts(wins_by_preset: dict):
    """Validate prefill win counts match the blog."""
    wins = wins_by_preset["prefill"]
    expected = {
        "max-autotune-no-cudagraphs": 17,
        "default": 4,
        "max-autotune": 1,
        "eager": 1,
    }
    for mode, count in expected.items():
        actual = len(wins.get(mode, []))
        assert actual == count, (
            f"Prefill {mode}: expected {count} wins, got {actual}"
        )
    assert len(wins.get("reduce-overhead", [])) == 0


def test_prefill_winning_workloads(wins_by_preset: dict, blog_text: str):
    """Validate the exact workload lists in the prefill wins table."""
    wins = wins_by_preset["prefill"]

    expected_mancg = sorted([
        "diff_cfg_blend", "diff_time_embedding_mlp", "diff_unet_resblock",
        "diff_vae_decoder_upsample", "llm_gelu_mlp", "llm_kl_divergence_loss",
        "llm_label_smoothed_ce_loss", "llm_layernorm",
        "llm_logits_projection_softmax", "llm_moe_router", "llm_qk_norm",
        "llm_rmsnorm", "llm_swiglu_mlp", "rl_ppo_clipped_loss",
        "vlm_patch_embedding_conv2d", "vlm_vision_rmsnorm_pool",
        "vlm_vit_mlp_block",
    ])
    expected_default = sorted([
        "diff_scheduler_step", "llm_rope_apply", "rl_dpo_loss", "rl_grpo_loss",
    ])
    expected_ma = sorted(["vlm_contrastive_nce_loss"])
    expected_eager = sorted(["llm_topk_sampling_prep"])

    assert wins.get("max-autotune-no-cudagraphs", []) == expected_mancg
    assert wins.get("default", []) == expected_default
    assert wins.get("max-autotune", []) == expected_ma
    assert wins.get("eager", []) == expected_eager

    # Also verify these lists appear in the blog
    for wl in expected_mancg:
        assert f"`{wl}`" in blog_text, f"Missing {wl} in blog prefill mancg row"
    for wl in expected_default:
        assert f"`{wl}`" in blog_text, f"Missing {wl} in blog prefill default row"


# ── Decode wins ──────────────────────────────────────────────────────


def test_decode_win_counts(wins_by_preset: dict):
    """Validate decode win counts match the blog (after exclusions)."""
    wins = wins_by_preset["decode"]
    expected = {
        "default": 10,
        "max-autotune-no-cudagraphs": 8,
    }
    for mode, count in expected.items():
        actual = len(wins.get(mode, []))
        assert actual == count, (
            f"Decode {mode}: expected {count} wins, got {actual}"
        )
    assert len(wins.get("max-autotune", [])) == 0
    assert len(wins.get("reduce-overhead", [])) == 0
    assert len(wins.get("eager", [])) == 0


def test_decode_winning_workloads(wins_by_preset: dict, blog_text: str):
    """Validate the exact workload lists in the decode wins table."""
    wins = wins_by_preset["decode"]

    expected_default = sorted([
        "diff_cfg_blend", "diff_scheduler_step", "diff_unet_resblock",
        "llm_gelu_mlp", "llm_kl_divergence_loss", "llm_layernorm",
        "llm_moe_router", "llm_rope_apply", "llm_topk_sampling_prep",
        "rl_ppo_clipped_loss",
    ])
    expected_mancg = sorted([
        "llm_label_smoothed_ce_loss", "llm_logits_projection_softmax",
        "llm_qk_norm", "llm_rmsnorm", "llm_swiglu_mlp",
        "vlm_patch_embedding_conv2d", "vlm_vision_rmsnorm_pool",
        "vlm_vit_mlp_block",
    ])

    assert wins.get("default", []) == expected_default
    assert wins.get("max-autotune-no-cudagraphs", []) == expected_mancg


def test_decode_excluded_workloads_not_in_results(blog_text: str):
    """Excluded workloads should not appear in the decode wins table."""
    decode_section = blog_text.split("#### Decode workloads")[1].split(
        "## Conclusion"
    )[0]
    # Extract just the wins table rows (between the table header and the
    # geomean section)
    table_section = decode_section.split("Geometric mean")[0]
    for wl in DECODE_EXCLUDED:
        assert f"`{wl}`" not in table_section, (
            f"Excluded workload {wl} found in decode wins table"
        )


# ── Geomean speedup values ───────────────────────────────────────────


@pytest.mark.parametrize("preset", ["prefill", "decode"])
def test_geomean_mancg_is_best(preset: str, gmean_by_preset: dict):
    """max-autotune-no-cudagraphs should have the highest geomean speedup."""
    gmean = gmean_by_preset[preset]
    best_mode = max(gmean, key=gmean.get)
    assert best_mode == "max-autotune-no-cudagraphs", (
        f"{preset}: best geomean mode is {best_mode}, "
        f"expected max-autotune-no-cudagraphs"
    )


@pytest.mark.parametrize("preset", ["prefill", "decode"])
def test_geomean_default_is_second(preset: str, gmean_by_preset: dict):
    """`default` should have the second-highest geomean speedup."""
    gmean = gmean_by_preset[preset]
    ranked = sorted(gmean.items(), key=lambda x: -x[1])
    assert ranked[1][0] == "default", (
        f"{preset}: second-best geomean mode is {ranked[1][0]}, expected default"
    )


@pytest.mark.parametrize("preset", ["prefill", "decode"])
def test_geomean_eager_is_baseline(preset: str, gmean_by_preset: dict):
    """`eager` geomean speedup should be ~1.0."""
    gmean = gmean_by_preset[preset]
    assert abs(gmean["eager"] - 1.0) < 0.01, (
        f"{preset}: eager geomean is {gmean['eager']:.4f}, expected ~1.0"
    )


def test_prefill_geomean_values(gmean_by_preset: dict):
    """Validate specific prefill geomean values match the blog plots."""
    g = gmean_by_preset["prefill"]
    assert abs(g["max-autotune-no-cudagraphs"] - 2.000) < 0.01
    assert abs(g["default"] - 1.874) < 0.01
    assert abs(g["max-autotune"] - 1.254) < 0.01
    assert abs(g["reduce-overhead"] - 1.224) < 0.01


def test_decode_geomean_values(gmean_by_preset: dict):
    """Validate specific decode geomean values match the blog plots."""
    g = gmean_by_preset["decode"]
    assert abs(g["max-autotune-no-cudagraphs"] - 1.687) < 0.01
    assert abs(g["default"] - 1.682) < 0.01
    assert abs(g["reduce-overhead"] - 0.862) < 0.01
    assert abs(g["max-autotune"] - 0.859) < 0.01


def test_decode_cudagraph_modes_below_eager(gmean_by_preset: dict):
    """For decode, reduce-overhead and max-autotune should be below 1.0x."""
    g = gmean_by_preset["decode"]
    assert g["reduce-overhead"] < 1.0
    assert g["max-autotune"] < 1.0


# ── Win margins ──────────────────────────────────────────────────────


def test_prefill_win_margins(win_margins_by_preset: dict):
    """Validate prefill win margin percentages from the blog."""
    m = win_margins_by_preset["prefill"]
    # h2h count includes workloads won overall by eager/max-autotune
    assert m["max-autotune-no-cudagraphs"]["wins"] == 19
    assert abs(m["max-autotune-no-cudagraphs"]["avg_margin"] - 7.58) < 0.1
    assert m["default"]["wins"] == 4
    assert abs(m["default"]["avg_margin"] - 2.45) < 0.1


def test_decode_win_margins(win_margins_by_preset: dict):
    """Validate decode win margin percentages from the blog."""
    m = win_margins_by_preset["decode"]
    assert m["default"]["wins"] == 10
    assert abs(m["default"]["avg_margin"] - 5.94) < 0.1
    assert m["max-autotune-no-cudagraphs"]["wins"] == 8
    assert abs(m["max-autotune-no-cudagraphs"]["avg_margin"] - 6.83) < 0.1


def test_prefill_margin_blog_table_values(blog_text: str):
    """Check the exact numbers that appear in the prefill win margin table."""
    assert "**7.58%**" in blog_text
    assert "2.45%" in blog_text


def test_decode_margin_blog_table_values(blog_text: str):
    """Check the exact numbers that appear in the decode win margin table."""
    assert "**6.83%**" in blog_text
    assert "5.94%" in blog_text


# ── Win margin table win counts match blog ───────────────────────────


def test_prefill_margin_table_win_count(blog_text: str):
    """The prefill win margin table should show 17 for mancg and 4 for default."""
    prefill_section = blog_text.split("#### Decode workloads")[0]
    assert "| `max-autotune-no-cudagraphs` | 17 | **7.58%**" in prefill_section
    assert "| `default` | 4 | 2.45%" in prefill_section


def test_decode_margin_table_win_count(blog_text: str):
    """The decode win margin table should show 8 for mancg and 10 for default."""
    decode_section = blog_text.split("#### Decode workloads")[1]
    assert "| `default` | 10 | 5.94%" in decode_section
    assert "| `max-autotune-no-cudagraphs` | 8 | **6.83%**" in decode_section


# ── Domain wins (validates the domain bar charts) ────────────────────


def test_prefill_domain_wins(domain_wins_by_preset: dict):
    """Validate prefill domain win counts match the generated chart."""
    d = domain_wins_by_preset["prefill"]
    # LLM 1/9, VLM 0/3, Diffusion 1/4, RL 2/1
    assert d.get("default", {}).get("llm", 0) == 1
    assert d.get("max-autotune-no-cudagraphs", {}).get("llm", 0) == 9
    assert d.get("default", {}).get("vlm", 0) == 0
    assert d.get("max-autotune-no-cudagraphs", {}).get("vlm", 0) == 3
    assert d.get("default", {}).get("diffusion", 0) == 1
    assert d.get("max-autotune-no-cudagraphs", {}).get("diffusion", 0) == 4
    assert d.get("default", {}).get("rl", 0) == 2
    assert d.get("max-autotune-no-cudagraphs", {}).get("rl", 0) == 1


def test_decode_domain_wins(domain_wins_by_preset: dict):
    """Validate decode domain win counts match the generated chart."""
    d = domain_wins_by_preset["decode"]
    # LLM 6/5, VLM 0/3, Diffusion 3/0, RL 1/0
    assert d.get("default", {}).get("llm", 0) == 6
    assert d.get("max-autotune-no-cudagraphs", {}).get("llm", 0) == 5
    assert d.get("default", {}).get("vlm", 0) == 0
    assert d.get("max-autotune-no-cudagraphs", {}).get("vlm", 0) == 3
    assert d.get("default", {}).get("diffusion", 0) == 3
    assert d.get("max-autotune-no-cudagraphs", {}).get("diffusion", 0) == 0
    assert d.get("default", {}).get("rl", 0) == 1
    assert d.get("max-autotune-no-cudagraphs", {}).get("rl", 0) == 0


# ── Prose claims ─────────────────────────────────────────────────────


def test_blog_claims_23_prefill_workloads(blog_text: str):
    """Blog says 'Out of 23 prefill workloads'."""
    assert "Out of 23 prefill workloads" in blog_text


def test_blog_claims_18_decode_workloads(blog_text: str):
    """Blog says '18 decode workloads'."""
    assert "18 decode workloads" in blog_text


def test_blog_prefill_prose_win_counts(blog_text: str):
    """Validate the prefill prose summary matches the table."""
    prefill_section = blog_text.split("#### Decode workloads")[0]
    assert "wins **17**" in prefill_section
    assert "wins **4**" in prefill_section
    assert "wins **1**" in prefill_section


def test_blog_decode_prose_win_counts(blog_text: str):
    """Validate the decode prose summary matches the table."""
    decode_section = blog_text.split("#### Decode workloads")[1].split(
        "## Conclusion"
    )[0]
    assert "wins **10**" in decode_section
    assert "wins **8**" in decode_section


def test_blog_reduce_overhead_wins_none_prefill(blog_text: str):
    """Blog claims reduce-overhead wins none in prefill."""
    prefill_section = blog_text.split("#### Decode workloads")[0]
    assert "`reduce-overhead` wins none" in prefill_section


def test_blog_no_ma_ro_eager_wins_decode(blog_text: str):
    """Blog claims neither max-autotune, reduce-overhead, nor eager wins decode."""
    decode_section = blog_text.split("#### Decode workloads")[1].split(
        "## Conclusion"
    )[0]
    assert "Neither `max-autotune`, `reduce-overhead`, nor `eager` wins any" in (
        decode_section
    )


# ── Decode exclusion documented ──────────────────────────────────────


def test_blog_documents_decode_exclusions(blog_text: str):
    """Blog should explain which workloads are excluded from decode."""
    for wl in DECODE_EXCLUDED:
        assert f"`{wl}`" in blog_text, (
            f"Excluded workload {wl} not mentioned in blog"
        )
    assert "invariant" in blog_text.lower() or "training-only" in blog_text.lower()


# ── Appendix shape presets match the study script ────────────────────


def test_appendix_shape_presets(blog_text: str):
    """Verify key shape parameters in the appendix match the study script."""
    appendix = blog_text.split("### Shape Presets")[1]
    assert "64" in appendix  # batch size decode
    assert "1" in appendix  # seq length decode
    assert "2048" in appendix  # hidden dim
    assert "16" in appendix  # attention heads
    assert "128" in appendix  # head dim
    assert "5504" in appendix  # intermediate size
    assert "32768" in appendix  # vocab size


def test_appendix_workload_count(blog_text: str):
    """Appendix workload descriptions table should have 23 entries."""
    appendix = blog_text.split("### Workload Descriptions")[1].split(
        "### Shape Presets"
    )[0]
    rows = [
        line
        for line in appendix.strip().split("\n")
        if line.startswith("| `")
    ]
    assert len(rows) == 23, f"Expected 23 workload descriptions, got {len(rows)}"


def test_appendix_workloads_match_csv(raw_df: pd.DataFrame, blog_text: str):
    """Every workload in the CSV should appear in the appendix."""
    appendix = blog_text.split("### Workload Descriptions")[1].split(
        "### Shape Presets"
    )[0]
    csv_workloads = sorted(raw_df["workload"].unique())
    for wl in csv_workloads:
        assert f"`{wl}`" in appendix, f"Workload {wl} missing from appendix"
