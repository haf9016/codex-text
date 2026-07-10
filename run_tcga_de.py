#!/usr/bin/env python3
"""Matched TCGA-GBM mutation/expression differential-expression analysis.

Public data source: UCSC Xena TCGA hub.
Primary comparison: operational IDH-wild-type primary GBMs with >=1 non-silent
mutation in any of the 25 genes explicitly listed in the prior report versus
operational IDH-wild-type primary GBMs without such a mutation.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path.cwd()
DATA = ROOT / "downloaded_data"
OUT = ROOT / "results"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

URLS = {
    "expr": [
        "https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/HiSeqV2.gz",
        "https://tcga.xenahubs.net/download/TCGA.GBMLGG.sampleMap/HiSeqV2.gz",
    ],
    "mut": [
        "https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/mutation.gz",
        "https://tcga.xenahubs.net/download/TCGA.GBMLGG.sampleMap/mutation.gz",
    ],
    "clin": [
        "https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/GBM_clinicalMatrix",
        "https://tcga.xenahubs.net/download/TCGA.GBM.sampleMap/GBM_clinicalMatrix.gz",
        "https://tcga.xenahubs.net/download/TCGA.GBMLGG.sampleMap/GBMLGG_clinicalMatrix.gz",
    ],
}

FOCUSED = [
    "SF3B1", "SRSF2", "U2AF1", "ZRSR2", "PRPF8", "LUC7L2", "RBM10",
    "FUBP1", "QKI", "HNRNPK", "DDX41", "PCBP1", "RBM5", "RBM6", "SF1",
    "U2AF2", "SRSF1", "TRA2B",
]
EXPANDED_ADDITIONS = [
    "PTBP1", "SRRM2", "RBM25", "RBM47", "HNRNPU", "HNRNPL", "HNRNPA2B1",
]
EXPANDED = FOCUSED + EXPANDED_ADDITIONS
CANONICAL15 = [
    "SF3B1", "SF3A1", "U2AF1", "U2AF2", "SRSF2", "ZRSR2", "RBM10",
    "PRPF8", "LUC7L2", "DDX3X", "SUGP1", "PHF5A", "RBM39", "HNRNPK",
    "FUBP1",
]
PANELS = {
    "expanded_explicit25": EXPANDED,
    "focused_explicit18": FOCUSED,
    "canonical15_sensitivity": CANONICAL15,
}


def download_first(name: str, urls: list[str]) -> tuple[Path, str]:
    suffix = ".gz" if any(u.endswith(".gz") for u in urls) else ".txt"
    dest = DATA / f"{name}{suffix}"
    last_error = None
    for url in urls:
        try:
            print(f"Downloading {name}: {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as w:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    w.write(chunk)
            if dest.stat().st_size < 100:
                raise RuntimeError(f"Downloaded file too small ({dest.stat().st_size} bytes)")
            print(f"  saved {dest} ({dest.stat().st_size:,} bytes)", flush=True)
            return dest, url
        except Exception as exc:
            last_error = exc
            print(f"  failed: {exc}", flush=True)
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"Unable to download {name}: {last_error}")


def read_matrix(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    df = pd.read_csv(path, sep="\t", index_col=0, compression=compression, low_memory=False)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def sample_type_code(sample: str) -> str | None:
    m = re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-([0-9]{2})", str(sample), flags=re.I)
    return m.group(1) if m else None


def patient_id(sample: str) -> str:
    s = str(sample)
    return s[:12] if s.startswith("TCGA-") and len(s) >= 12 else s


def choose_expression_samples(columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        code = sample_type_code(col)
        if code is None or code == "01":
            rows.append((patient_id(col), col, code or "patient_level"))
    d = pd.DataFrame(rows, columns=["patient_id", "expr_sample_id", "sample_type_code"])
    if d.empty:
        raise RuntimeError("No primary/patient-level TCGA expression samples identified")
    return d.sort_values(["patient_id", "expr_sample_id"]).drop_duplicates("patient_id").reset_index(drop=True)


def aggregate_mutation_by_patient(mut: pd.DataFrame, patients: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    mut_num = (mut.apply(pd.to_numeric, errors="coerce").fillna(0.0) != 0).astype(np.int8)
    col_by_patient: dict[str, list[str]] = {}
    for c in mut_num.columns:
        col_by_patient.setdefault(patient_id(c), []).append(c)
    out, source_cols = {}, {}
    for p in patients:
        cols = col_by_patient.get(p, [])
        if not cols:
            continue
        primary = [c for c in cols if sample_type_code(c) in (None, "01")]
        use = primary if primary else cols
        out[p] = mut_num[use].max(axis=1)
        source_cols[p] = ";".join(use)
    if not out:
        raise RuntimeError("No overlap between expression patients and mutation matrix")
    return pd.DataFrame(out), source_cols


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    po = p[ok]
    order = np.argsort(po)
    ranked = po[order]
    n = len(ranked)
    q = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    out[ok] = np.clip(q[inv], 0, 1)
    return out


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / math.sqrt(pooled)
    return d * (1 - 3 / max(4 * (nx + ny) - 9, 1))


def run_de(expr: pd.DataFrame, group: pd.Series, label: str) -> pd.DataFrame:
    g = group.astype(bool)
    mut_cols, wt_cols = group.index[g], group.index[~g]
    if len(mut_cols) < 3 or len(wt_cols) < 3:
        raise RuntimeError(f"Insufficient groups for {label}: {len(mut_cols)} vs {len(wt_cols)}")
    rows = []
    for gene, row in expr.iterrows():
        xm = pd.to_numeric(row.loc[mut_cols], errors="coerce").dropna().to_numpy(float)
        xw = pd.to_numeric(row.loc[wt_cols], errors="coerce").dropna().to_numpy(float)
        allx = np.concatenate([xm, xw])
        expressed_fraction = float(np.mean(allx > 0.1)) if len(allx) else np.nan
        sd_all = float(np.std(allx, ddof=1)) if len(allx) > 1 else np.nan
        tested = len(xm) >= 3 and len(xw) >= 3 and np.isfinite(sd_all) and sd_all > 0.1 and expressed_fraction >= 0.20
        if tested:
            tt = stats.ttest_ind(xm, xw, equal_var=False, nan_policy="omit")
            p_welch, t_stat = float(tt.pvalue), float(tt.statistic)
            try:
                p_mw = float(stats.mannwhitneyu(xm, xw, alternative="two-sided").pvalue)
            except Exception:
                p_mw = np.nan
        else:
            p_welch = t_stat = p_mw = np.nan
        rows.append({
            "gene_symbol": str(gene), "n_mut": len(xm), "n_wt": len(xw),
            "mean_log2_expr_mut": float(np.mean(xm)) if len(xm) else np.nan,
            "mean_log2_expr_wt": float(np.mean(xw)) if len(xw) else np.nan,
            "median_log2_expr_mut": float(np.median(xm)) if len(xm) else np.nan,
            "median_log2_expr_wt": float(np.median(xw)) if len(xw) else np.nan,
            "log2_fold_change": float(np.mean(xm) - np.mean(xw)),
            "hedges_g": hedges_g(xm, xw), "welch_t": t_stat,
            "p_value_welch": p_welch, "p_value_mann_whitney": p_mw,
            "expressed_fraction": expressed_fraction, "sd_all": sd_all, "tested": bool(tested),
        })
    res = pd.DataFrame(rows)
    res["fdr_bh"] = bh_fdr(res["p_value_welch"].to_numpy())
    res["significant_fdr05"] = (res["fdr_bh"] < 0.05) & (res["log2_fold_change"].abs() >= 0.5)
    res["nominal_p05"] = (res["p_value_welch"] < 0.05) & (res["log2_fold_change"].abs() >= 0.5)
    res["rank_score"] = np.sign(res["log2_fold_change"]) * -np.log10(res["p_value_welch"].clip(lower=1e-300))
    return res.sort_values(["fdr_bh", "p_value_welch", "gene_symbol"], na_position="last").reset_index(drop=True)


def volcano(res: pd.DataFrame, title: str, path: Path) -> None:
    d = res[res["tested"] & res["p_value_welch"].notna()].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    y = -np.log10(d["p_value_welch"].clip(lower=1e-300))
    ax.scatter(d["log2_fold_change"], y, s=10, alpha=0.55)
    ax.axvline(-0.5, linewidth=0.8, linestyle="--")
    ax.axvline(0.5, linewidth=0.8, linestyle="--")
    ax.axhline(-math.log10(0.05), linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean difference on log2(RSEM normalized count + 1) scale")
    ax.set_ylabel("-log10 Welch p-value")
    ax.set_title(title)
    for _, r in d.nsmallest(12, "p_value_welch").iterrows():
        ax.annotate(r["gene_symbol"], (r["log2_fold_change"], -math.log10(max(r["p_value_welch"], 1e-300))), fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def pca_plot(expr: pd.DataFrame, group: pd.Series, path: Path) -> None:
    top = expr.loc[expr.var(axis=1).nlargest(min(2000, len(expr))).index].T
    x = np.nan_to_num(top.to_numpy(float) - np.nanmean(top.to_numpy(float), axis=0), nan=0.0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    score = u[:, :2] * s[:2]
    var_exp = (s ** 2) / np.sum(s ** 2)
    fig, ax = plt.subplots(figsize=(7, 6))
    for status, marker in [(False, "o"), (True, "^")]:
        mask = group.loc[top.index].astype(bool).to_numpy() == status
        ax.scatter(score[mask, 0], score[mask, 1], s=35, alpha=0.7, marker=marker, label="Panel WT" if not status else "Panel mutated")
    ax.set_xlabel(f"PC1 ({100*var_exp[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100*var_exp[1]:.1f}%)")
    ax.set_title("TCGA IDH-wild-type GBM RNA-seq PCA"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main() -> None:
    provenance = {}
    expr_path, provenance["expression_url"] = download_first("expression", URLS["expr"])
    mut_path, provenance["mutation_url"] = download_first("mutation", URLS["mut"])
    try:
        _, provenance["clinical_url"] = download_first("clinical", URLS["clin"])
    except Exception as exc:
        provenance["clinical_download_error"] = str(exc)

    expr_raw, mut_raw = read_matrix(expr_path), read_matrix(mut_path)
    print("Expression shape:", expr_raw.shape, flush=True)
    print("Mutation shape:", mut_raw.shape, flush=True)
    expr_samples_all = choose_expression_samples(list(expr_raw.columns))
    expr_num = expr_raw.apply(pd.to_numeric, errors="coerce").groupby(expr_raw.index).mean()
    mut_patient, mut_source_cols = aggregate_mutation_by_patient(mut_raw, expr_samples_all["patient_id"].tolist())
    common_patients = sorted(set(expr_samples_all["patient_id"]) & set(mut_patient.columns))
    expr_samples = expr_samples_all[expr_samples_all["patient_id"].isin(common_patients)].sort_values("patient_id")
    expr_by_patient = pd.DataFrame({r["patient_id"]: expr_num[r["expr_sample_id"]] for _, r in expr_samples.iterrows()})
    expr_by_patient = expr_by_patient.loc[:, common_patients]
    mut_patient = mut_patient.loc[:, common_patients]

    idh_rows = [g for g in ["IDH1", "IDH2"] if g in mut_patient.index]
    idh_mut = mut_patient.loc[idh_rows].max(axis=0).astype(bool) if idh_rows else pd.Series(False, index=common_patients)
    idh_wt_patients = idh_mut.index[~idh_mut].tolist()
    expr_idhwt, mut_idhwt = expr_by_patient.loc[:, idh_wt_patients], mut_patient.loc[:, idh_wt_patients]
    burden = mut_idhwt.sum(axis=0).astype(int)

    sample_meta = expr_samples.set_index("patient_id").loc[idh_wt_patients].reset_index()
    sample_meta["mutation_source_columns"] = sample_meta["patient_id"].map(mut_source_cols)
    sample_meta["idh1_or_idh2_nonsilent_mutation"] = False
    sample_meta["nonsilent_mutated_gene_count"] = sample_meta["patient_id"].map(burden)

    panel_records = []
    summary = {
        "data_provenance": provenance, "raw_expression_shape": list(expr_raw.shape),
        "raw_mutation_shape": list(mut_raw.shape),
        "n_expression_primary_or_patient_level": int(len(expr_samples_all)),
        "n_matched_expression_mutation_patients": int(len(common_patients)),
        "n_operational_idh_wildtype": int(len(idh_wt_patients)),
        "n_operational_idh_mutant_excluded": int(idh_mut.sum()),
        "idh_mutant_patient_ids": sorted(idh_mut.index[idh_mut].tolist()),
        "expression_scale": "UCSC Xena HiSeqV2: log2(RSEM normalized count + 1)",
        "mutation_definition": "UCSC Xena PANCAN AWG gene-level non-silent mutation matrix; any non-zero call",
        "primary_sample_rule": "TCGA sample type 01; one lexicographically first expression sample per patient",
        "de_method": "Welch two-sample t-test on log2 expression; Benjamini-Hochberg FDR; tested genes required SD > 0.1 and expression > 0.1 in >=20% of samples",
        "panels": {},
    }

    for label, panel in PANELS.items():
        present, absent = [g for g in panel if g in mut_idhwt.index], [g for g in panel if g not in mut_idhwt.index]
        group = mut_idhwt.loc[present].max(axis=0).astype(bool) if present else pd.Series(False, index=idh_wt_patients)
        sample_meta[f"group_{label}"] = sample_meta["patient_id"].map(group).fillna(False).astype(bool)
        sample_meta[f"mutated_panel_genes_{label}"] = sample_meta["patient_id"].map(lambda p: ";".join([g for g in present if bool(mut_idhwt.at[g, p])]))
        for p in idh_wt_patients:
            for g in [g for g in present if bool(mut_idhwt.at[g, p])]:
                panel_records.append({"analysis": label, "patient_id": p, "gene": g})
        res = run_de(expr_idhwt, group, label)
        res.to_csv(OUT / f"de_{label}_all.csv", index=False)
        res.head(1000).to_csv(OUT / f"de_{label}_top1000.csv", index=False)
        res[res["significant_fdr05"]].to_csv(OUT / f"de_{label}_fdr05_fc05.csv", index=False)
        res[res["nominal_p05"]].to_csv(OUT / f"de_{label}_nominal_p05_fc05.csv", index=False)
        volcano(res, f"{label}: panel-mutated vs panel-WT IDH-wild-type GBM", OUT / f"volcano_{label}.png")
        if label == "expanded_explicit25": pca_plot(expr_idhwt, group, OUT / "pca_expanded_explicit25.png")
        summary["panels"][label] = {
            "genes_requested": panel, "genes_with_rows_in_mutation_matrix": present,
            "genes_without_rows_no_nonsilent_calls": absent, "n_mutated": int(group.sum()),
            "n_wildtype": int((~group).sum()), "mutated_patient_ids": sorted(group.index[group].tolist()),
            "n_genes_tested": int(res["tested"].sum()),
            "n_fdr05_and_abs_log2fc_ge_0_5": int(res["significant_fdr05"].sum()),
            "n_nominal_p05_and_abs_log2fc_ge_0_5": int(res["nominal_p05"].sum()),
            "top20": res.head(20)[["gene_symbol", "log2_fold_change", "p_value_welch", "fdr_bh"]].to_dict("records"),
        }

    sample_meta.to_csv(OUT / "cohort_sample_metadata.csv", index=False)
    pd.DataFrame(panel_records).to_csv(OUT / "panel_mutation_events.csv", index=False)
    pd.DataFrame([{"analysis": k, "panel_gene": g, "panel_membership": "focused" if g in FOCUSED else ("expanded_addition" if g in EXPANDED_ADDITIONS else "canonical15_only")} for k, genes in PANELS.items() for g in genes]).to_csv(OUT / "panel_definitions.csv", index=False)
    q95 = float(burden.quantile(0.95)) if len(burden) else np.nan
    sample_meta2 = pd.read_csv(OUT / "cohort_sample_metadata.csv")
    sample_meta2["hypermutation_q95_flag"] = sample_meta2["nonsilent_mutated_gene_count"] > q95
    sample_meta2.to_csv(OUT / "cohort_sample_metadata.csv", index=False)
    summary["mutation_burden_q95"] = q95
    summary["hypermutation_q95_patient_ids"] = sample_meta2.loc[sample_meta2["hypermutation_q95_flag"], "patient_id"].tolist()
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "README.txt").write_text(
        "TCGA IDH-wild-type GBM differential expression analysis\n\n"
        "Primary comparison: mutation in any of the 25 genes explicitly listed in the prior report versus no mutation in those genes.\n"
        "Mutation calls: UCSC Xena PANCAN AWG gene-level non-silent matrix.\n"
        "Expression: UCSC Xena HiSeqV2 log2(RSEM normalized count + 1).\n"
        "Statistics: Welch t-test with BH FDR; exploratory because the mutated group is small and heterogeneous.\n"
        "The report headings said 19/26 genes, but the explicit table contains 18 focused + 7 additions = 25 unique genes. The analysis uses the explicit names rather than inventing a missing gene.\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
