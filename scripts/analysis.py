from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = "https://github.com/cBioPortal/datahub/raw/refs/heads/master/public/gbm_tcga_pan_can_atlas_2018"
OUT = Path("out")
WORK = Path("work")
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)

FILES = {
    "expression": f"{BASE}/data_mrna_seq_v2_rsem.txt",
    "mutations": f"{BASE}/data_mutations.txt",
    "clinical_sample": f"{BASE}/data_clinical_sample.txt",
    "clinical_patient": f"{BASE}/data_clinical_patient.txt",
    "cases_sequenced": f"{BASE}/case_lists/cases_sequenced.txt",
}

FOCUSED_GENES = [
    "SF3B1", "SRSF2", "U2AF1", "ZRSR2", "PRPF8", "LUC7L2", "RBM10", "FUBP1",
    "QKI", "HNRNPK", "DDX41", "PCBP1", "RBM5", "RBM6", "SF1", "U2AF2", "SRSF1",
    "TRA2B",
]
EXPANDED_GENES = FOCUSED_GENES + [
    "PTBP1", "SRRM2", "RBM25", "RBM47", "HNRNPU", "HNRNPL", "HNRNPA2B1",
]
IDH_GENES = ["IDH1", "IDH2"]

FUNCTIONAL_VARIANT_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Translation_Start_Site",
    "Start_Codon_SNP", "Start_Codon_Del", "Start_Codon_Ins",
    "Stop_Codon_Del", "Stop_Codon_Ins",
    "De_novo_Start_InFrame", "De_novo_Start_OutOfFrame",
    "Multi_Hit",
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        return
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=240) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")


def barcode15(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    return x[:15]


def patient12(x: str) -> str:
    return barcode15(x)[:12]


def sample_type_code(x: str) -> str:
    b = barcode15(x)
    try:
        return b.split("-")[3]
    except Exception:
        return ""


def is_primary_tumor(x: str) -> bool:
    return sample_type_code(x) == "01"


def parse_case_list(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size < 10:
        return set()
    cases = set()
    for line in path.read_text().splitlines():
        if line.startswith("case_list_ids:"):
            raw = line.split(":", 1)[1].strip()
            cases.update(barcode15(x) for x in raw.replace(",", "\t").split() if x.strip())
    return cases


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan, dtype=float)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / math.sqrt(pooled)


def run_de(expr_log2: pd.DataFrame, group_series: pd.Series, label: str) -> pd.DataFrame:
    samples = [s for s in expr_log2.columns if s in group_series.index and pd.notna(group_series.loc[s])]
    expr = expr_log2[samples]
    g = group_series.loc[samples].astype(bool)
    mutated = list(g[g].index)
    unmutated = list(g[~g].index)
    rows = []
    for gene, values in expr.iterrows():
        vals = values.astype(float).values
        if np.nanvar(vals) < 1e-8:
            continue
        pct_expr = float(np.mean(vals > 1.0))
        if pct_expr < 0.10:
            continue
        a = values[mutated].astype(float).values
        b = values[unmutated].astype(float).values
        try:
            t_res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            p = float(t_res.pvalue)
        except Exception:
            p = np.nan
        mean_a = float(np.nanmean(a))
        mean_b = float(np.nanmean(b))
        log2fc = mean_a - mean_b
        rows.append({
            "gene": gene,
            "comparison": label,
            "n_mutated": int(len(mutated)),
            "n_unmutated": int(len(unmutated)),
            "mean_mutated_log2": mean_a,
            "mean_unmutated_log2": mean_b,
            "median_mutated_log2": float(np.nanmedian(a)),
            "median_unmutated_log2": float(np.nanmedian(b)),
            "log2FC_mutated_vs_unmutated": log2fc,
            "p_value": p,
            "cohens_d": cohens_d(a, b),
            "pct_samples_expressed": pct_expr,
            "direction": "higher in mutated" if log2fc > 0 else "lower in mutated",
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["FDR_BH"] = bh_fdr(res["p_value"].values)
    res["abs_log2FC"] = res["log2FC_mutated_vs_unmutated"].abs()
    res = res.sort_values(["FDR_BH", "p_value", "abs_log2FC"], ascending=[True, True, False]).reset_index(drop=True)
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    return res


for key, url in FILES.items():
    download_file(url, WORK / f"{key}.txt")

# Expression matrix.
expr_raw = pd.read_csv(WORK / "expression.txt", sep="\t", low_memory=False)
gene_col = "Hugo_Symbol" if "Hugo_Symbol" in expr_raw.columns else expr_raw.columns[0]
drop_cols = [c for c in ["Hugo_Symbol", "Entrez_Gene_Id"] if c in expr_raw.columns]
sample_cols = [c for c in expr_raw.columns if c not in drop_cols]
expr_raw[gene_col] = expr_raw[gene_col].astype(str)
expr_data = expr_raw[[gene_col] + sample_cols].copy()
for c in sample_cols:
    expr_data[c] = pd.to_numeric(expr_data[c], errors="coerce")
expr_by_gene = expr_data.groupby(gene_col)[sample_cols].mean(numeric_only=True)
expr_by_gene = expr_by_gene[~expr_by_gene.index.str.startswith("?")]
expr_by_gene = expr_by_gene.rename(columns={c: barcode15(c) for c in expr_by_gene.columns})
expr_by_gene = expr_by_gene.loc[:, ~expr_by_gene.columns.duplicated()]
primary_expr_cols = [c for c in expr_by_gene.columns if is_primary_tumor(c)]
expr_by_gene = expr_by_gene[primary_expr_cols]
expr_log2 = np.log2(expr_by_gene.astype(float) + 1.0)

# Mutations.
mut = pd.read_csv(WORK / "mutations.txt", sep="\t", comment="#", low_memory=False)
if "Tumor_Sample_Barcode" not in mut.columns:
    if "SAMPLE_ID" in mut.columns:
        mut["Tumor_Sample_Barcode"] = mut["SAMPLE_ID"]
    else:
        raise RuntimeError("Mutation file lacks Tumor_Sample_Barcode and SAMPLE_ID")
mut["sample_id"] = mut["Tumor_Sample_Barcode"].map(barcode15)
mut["patient_id"] = mut["sample_id"].map(patient12)
mut["Variant_Classification"] = mut["Variant_Classification"].fillna("")
functional_mut = mut[mut["Variant_Classification"].isin(FUNCTIONAL_VARIANT_CLASSES)].copy()

sequenced = parse_case_list(WORK / "cases_sequenced.txt")
if not sequenced:
    sequenced = set(mut["sample_id"].unique())
expr_samples = set(expr_log2.columns)
primary_sequenced = {s for s in sequenced if is_primary_tumor(s)}
matched_samples = sorted(expr_samples & primary_sequenced)
if len(matched_samples) < 20:
    seq_patients = {patient12(s) for s in sequenced}
    matched_samples = sorted([s for s in expr_samples if is_primary_tumor(s) and patient12(s) in seq_patients])
if len(matched_samples) < 20:
    raise RuntimeError(f"Too few matched primary tumor samples: {len(matched_samples)}")

idh_mut_samples = set(functional_mut.loc[functional_mut["Hugo_Symbol"].isin(IDH_GENES), "sample_id"].unique())
idhwt_samples = sorted([s for s in matched_samples if s not in idh_mut_samples])

func_panel_mut = functional_mut[functional_mut["sample_id"].isin(idhwt_samples)].copy()
focused_mut = func_panel_mut[func_panel_mut["Hugo_Symbol"].isin(FOCUSED_GENES)]
expanded_mut = func_panel_mut[func_panel_mut["Hugo_Symbol"].isin(EXPANDED_GENES)]
focused_pos = set(focused_mut["sample_id"].unique())
expanded_pos = set(expanded_mut["sample_id"].unique())

sample_info = pd.DataFrame({"sample_id": idhwt_samples})
sample_info["patient_id"] = sample_info["sample_id"].map(patient12)
sample_info["focused_panel_mutated"] = sample_info["sample_id"].isin(focused_pos)
sample_info["expanded_panel_mutated"] = sample_info["sample_id"].isin(expanded_pos)
sample_info["functional_mutation_count_total"] = sample_info["sample_id"].map(functional_mut.groupby("sample_id").size()).fillna(0).astype(int)
sample_info["focused_mutated_genes"] = sample_info["sample_id"].map(lambda s: ";".join(sorted(focused_mut.loc[focused_mut["sample_id"] == s, "Hugo_Symbol"].unique())))
sample_info["expanded_mutated_genes"] = sample_info["sample_id"].map(lambda s: ";".join(sorted(expanded_mut.loc[expanded_mut["sample_id"] == s, "Hugo_Symbol"].unique())))

# Variant details.
detail_cols = ["sample_id", "Hugo_Symbol", "Variant_Classification", "Variant_Type", "Protein_Change", "HGVSp_Short", "HGVSc", "Start_Position", "End_Position"]
available_detail_cols = [c for c in detail_cols if c in mut.columns]
panel_variant_details = func_panel_mut.loc[func_panel_mut["Hugo_Symbol"].isin(EXPANDED_GENES), available_detail_cols].sort_values(["sample_id", "Hugo_Symbol"])

expr_matched = expr_log2[idhwt_samples]
focused_group = sample_info.set_index("sample_id")["focused_panel_mutated"]
expanded_group = sample_info.set_index("sample_id")["expanded_panel_mutated"]
focused_de = run_de(expr_matched, focused_group, "Focused 19-gene panel mutated vs unmutated")
expanded_de = run_de(expr_matched, expanded_group, "Expanded 26-gene panel mutated vs unmutated")

focused_sig = focused_de[(focused_de["FDR_BH"] < 0.05) & (focused_de["abs_log2FC"] >= 0.5)].copy() if not focused_de.empty else focused_de
expanded_sig = expanded_de[(expanded_de["FDR_BH"] < 0.05) & (expanded_de["abs_log2FC"] >= 0.5)].copy() if not expanded_de.empty else expanded_de

panel_rows = []
for g in FOCUSED_GENES:
    panel_rows.append({"panel": "Focused 19-gene", "gene": g, "included_in_focused": True})
for g in EXPANDED_GENES:
    panel_rows.append({"panel": "Expanded 26-gene", "gene": g, "included_in_focused": g in FOCUSED_GENES})
gene_panels = pd.DataFrame(panel_rows)

summary = {
    "source_study": "cBioPortal DataHub: gbm_tcga_pan_can_atlas_2018",
    "expression_file": FILES["expression"],
    "mutation_file": FILES["mutations"],
    "case_list_file": FILES["cases_sequenced"],
    "expression_transform": "log2(RSEM + 1)",
    "variant_grouping": "Functional/protein-altering and canonical splice-site somatic variants; Silent/Synonymous and noncoding calls excluded",
    "primary_matched_expression_and_mutation_samples": int(len(matched_samples)),
    "IDH_mutated_excluded": int(len(idh_mut_samples & set(matched_samples))),
    "IDHwt_primary_matched_samples": int(len(idhwt_samples)),
    "focused_mutated_n": int(sample_info["focused_panel_mutated"].sum()),
    "focused_unmutated_n": int((~sample_info["focused_panel_mutated"]).sum()),
    "expanded_mutated_n": int(sample_info["expanded_panel_mutated"].sum()),
    "expanded_unmutated_n": int((~sample_info["expanded_panel_mutated"]).sum()),
    "genes_tested_focused": int(len(focused_de)),
    "genes_tested_expanded": int(len(expanded_de)),
    "focused_FDR05_absLFC05_n": int(len(focused_sig)),
    "expanded_FDR05_absLFC05_n": int(len(expanded_sig)),
    "statistical_test": "Welch two-sample t-test on log2(RSEM+1), BH FDR",
}

def top_genes(df, n=20):
    if df.empty:
        return []
    return df[["rank", "gene", "log2FC_mutated_vs_unmutated", "p_value", "FDR_BH", "direction"]].head(n).to_dict("records")

summary["focused_top20"] = top_genes(focused_de)
summary["expanded_top20"] = top_genes(expanded_de)
summary["focused_sig_genes"] = focused_sig["gene"].head(200).tolist() if not focused_sig.empty else []
summary["expanded_sig_genes"] = expanded_sig["gene"].head(200).tolist() if not expanded_sig.empty else []

pd.DataFrame([summary]).to_csv(OUT / "analysis_summary.csv", index=False)
sample_info.to_csv(OUT / "sample_groups.csv", index=False)
gene_panels.to_csv(OUT / "gene_panels.csv", index=False)
panel_variant_details.to_csv(OUT / "panel_variant_details.csv", index=False)
focused_de.to_csv(OUT / "focused_DE_all_ranked.csv", index=False)
expanded_de.to_csv(OUT / "expanded_DE_all_ranked.csv", index=False)
focused_sig.to_csv(OUT / "focused_DE_FDR05_absLFC05.csv", index=False)
expanded_sig.to_csv(OUT / "expanded_DE_FDR05_absLFC05.csv", index=False)
focused_de.head(100).to_csv(OUT / "focused_DE_top100.csv", index=False)
expanded_de.head(100).to_csv(OUT / "expanded_DE_top100.csv", index=False)
(OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

notes = [
    "TCGA GBM splicing-factor mutation differential expression analysis",
    "",
    f"IDH-wild-type matched primary tumors: {len(idhwt_samples)}",
    f"Focused-panel mutated tumors: {int(sample_info['focused_panel_mutated'].sum())}",
    f"Focused-panel unmutated tumors: {int((~sample_info['focused_panel_mutated']).sum())}",
    f"Expanded-panel mutated tumors: {int(sample_info['expanded_panel_mutated'].sum())}",
    f"Expanded-panel unmutated tumors: {int((~sample_info['expanded_panel_mutated']).sum())}",
    "",
    f"Focused strict DEG count (FDR < 0.05 and |log2FC| >= 0.5): {len(focused_sig)}",
    f"Expanded strict DEG count (FDR < 0.05 and |log2FC| >= 0.5): {len(expanded_sig)}",
    "",
    "Positive log2FC indicates higher expression in the splicing-factor-mutated group.",
    "Negative log2FC indicates lower expression in the splicing-factor-mutated group.",
]
(OUT / "analysis_notes.txt").write_text("\n".join(notes))
print(json.dumps(summary, indent=2)[:4000])
