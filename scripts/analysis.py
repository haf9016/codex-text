from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

BASE = "https://github.com/cBioPortal/datahub/raw/refs/heads/master/public/gbm_tcga_pan_can_atlas_2018"
OUT = Path("out")
WORK = Path("work")
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)

FILES = {
    "mutations": f"{BASE}/data_mutations.txt",
    "clinical_patient": f"{BASE}/data_clinical_patient.txt",
    "clinical_sample": f"{BASE}/data_clinical_sample.txt",
    "cases_sequenced": f"{BASE}/case_lists/cases_sequenced.txt",
}

# Exact gene lists used in the preceding differential-expression analysis.
# The earlier documents called these 19- and 26-gene panels, but the enumerated
# lists contain 18 and 25 genes, respectively. This analysis uses the exact
# enumerated genes so that group assignments remain consistent.
FOCUSED_GENES = [
    "SF3B1", "SRSF2", "U2AF1", "ZRSR2", "PRPF8", "LUC7L2", "RBM10", "FUBP1",
    "QKI", "HNRNPK", "DDX41", "PCBP1", "RBM5", "RBM6", "SF1", "U2AF2", "SRSF1",
    "TRA2B",
]
EXPANDED_GENES = FOCUSED_GENES + [
    "PTBP1", "SRRM2", "RBM25", "RBM47", "HNRNPU", "HNRNPL", "HNRNPA2B1",
]
IDH_GENES = {"IDH1", "IDH2"}
FUNCTIONAL_VARIANT_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Translation_Start_Site",
    "Start_Codon_SNP", "Start_Codon_Del", "Start_Codon_Ins",
    "Stop_Codon_Del", "Stop_Codon_Ins",
    "De_novo_Start_InFrame", "De_novo_Start_OutOfFrame", "Multi_Hit",
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        return
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=240) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")


def barcode15(value: object) -> str:
    return str(value)[:15]


def patient12(value: object) -> str:
    return barcode15(value)[:12]


def sample_type_code(value: object) -> str:
    parts = barcode15(value).split("-")
    return parts[3] if len(parts) > 3 else ""


def is_primary_tumor(value: object) -> bool:
    return sample_type_code(value) == "01"


def parse_case_list(path: Path) -> set[str]:
    cases: set[str] = set()
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("case_list_ids:"):
            raw = line.split(":", 1)[1].strip()
            cases.update(barcode15(x) for x in raw.replace(",", "\t").split() if x.strip())
    return cases


def read_cbio_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    upper_to_actual = {column.upper(): column for column in columns}
    for candidate in candidates:
        if candidate.upper() in upper_to_actual:
            return upper_to_actual[candidate.upper()]
    return None


def parse_event(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().upper()
    if not text or text in {"NA", "NAN", "[NOT AVAILABLE]", "[NOT APPLICABLE]", "UNKNOWN"}:
        return np.nan
    if text.startswith("1:") or text == "1":
        return 1.0
    if text.startswith("0:") or text == "0":
        return 0.0
    event_words = ("DECEASED", "DEAD", "PROGRESSION", "PROGRESSED", "RECURRENCE", "RECURRED", "RELAPSE")
    censor_words = ("LIVING", "ALIVE", "CENSORED", "DISEASEFREE", "DISEASE FREE", "NO EVENT")
    if any(word in text for word in event_words):
        return 1.0
    if any(word in text for word in censor_words):
        return 0.0
    return np.nan


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({
        "[Not Available]": np.nan,
        "[Not Applicable]": np.nan,
        "[Discrepancy]": np.nan,
        "NA": np.nan,
        "": np.nan,
    }), errors="coerce")


def normalize_age(series: pd.Series) -> pd.Series:
    age = numeric_series(series)
    if age.dropna().median() > 200:
        age = age / 365.25
    return age


def km_object(time: np.ndarray, event: np.ndarray) -> SurvfuncRight | None:
    if len(time) == 0:
        return None
    return SurvfuncRight(time.astype(float), event.astype(int))


def loglog_ci(survival: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(survival) or not np.isfinite(se):
        return np.nan, np.nan
    if survival <= 0:
        return 0.0, 0.0
    if survival >= 1 or se <= 0:
        return survival, survival
    z = stats.norm.ppf(1 - alpha / 2)
    theta = math.log(-math.log(survival))
    se_theta = se / abs(survival * math.log(survival))
    lower = math.exp(-math.exp(theta + z * se_theta))
    upper = math.exp(-math.exp(theta - z * se_theta))
    return max(0.0, lower), min(1.0, upper)


def survival_at(sf: SurvfuncRight | None, month: float) -> tuple[float, float, float]:
    if sf is None or len(sf.surv_times) == 0:
        return 1.0, 1.0, 1.0
    index = np.searchsorted(sf.surv_times, month, side="right") - 1
    if index < 0:
        return 1.0, 1.0, 1.0
    value = float(sf.surv_prob[index])
    lower, upper = loglog_ci(value, float(sf.surv_prob_se[index]))
    return value, lower, upper


def finite_or_nan(value: object) -> float:
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def km_group_metrics(time: np.ndarray, event: np.ndarray) -> dict[str, float | int]:
    sf = km_object(time, event)
    median = np.nan
    median_low = np.nan
    median_high = np.nan
    if sf is not None:
        median = finite_or_nan(sf.quantile(0.5))
        try:
            median_low, median_high = sf.quantile_ci(0.5, method="cloglog")
            median_low = finite_or_nan(median_low)
            median_high = finite_or_nan(median_high)
        except Exception:
            pass
    s12, s12_low, s12_high = survival_at(sf, 12.0)
    s24, s24_low, s24_high = survival_at(sf, 24.0)
    return {
        "n": int(len(time)),
        "events": int(np.sum(event)),
        "censored": int(len(time) - np.sum(event)),
        "median_months": median,
        "median_ci_low": median_low,
        "median_ci_high": median_high,
        "survival_12m": s12,
        "survival_12m_ci_low": s12_low,
        "survival_12m_ci_high": s12_high,
        "survival_24m": s24,
        "survival_24m_ci_low": s24_low,
        "survival_24m_ci_high": s24_high,
    }


def rmst(time: np.ndarray, event: np.ndarray, tau: float = 24.0) -> float:
    sf = km_object(time, event)
    if sf is None:
        return np.nan
    area = 0.0
    previous = 0.0
    survival = 1.0
    for current, next_survival in zip(sf.surv_times, sf.surv_prob):
        current = float(current)
        if current >= tau:
            break
        area += (current - previous) * survival
        previous = current
        survival = float(next_survival)
    area += max(0.0, tau - previous) * survival
    return area


def bootstrap_rmst_difference(
    mutated_time: np.ndarray,
    mutated_event: np.ndarray,
    unmutated_time: np.ndarray,
    unmutated_event: np.ndarray,
    tau: float = 24.0,
    n_boot: int = 3000,
    seed: int = 1729,
) -> tuple[float, float, float, float]:
    observed = rmst(mutated_time, mutated_event, tau) - rmst(unmutated_time, unmutated_event, tau)
    if len(mutated_time) < 2 or len(unmutated_time) < 2:
        return observed, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    differences = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        mi = rng.integers(0, len(mutated_time), len(mutated_time))
        ui = rng.integers(0, len(unmutated_time), len(unmutated_time))
        differences[index] = (
            rmst(mutated_time[mi], mutated_event[mi], tau)
            - rmst(unmutated_time[ui], unmutated_event[ui], tau)
        )
    low, high = np.nanpercentile(differences, [2.5, 97.5])
    p_value = min(1.0, 2 * min(np.mean(differences <= 0), np.mean(differences >= 0)))
    return observed, float(low), float(high), float(p_value)


def cox_model(time: np.ndarray, event: np.ndarray, exog: np.ndarray) -> tuple[float, float, float, float]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = PHReg(time.astype(float), exog.astype(float), status=event.astype(int), ties="efron").fit(disp=0)
        beta = float(fit.params[0])
        se = float(fit.bse[0])
        hazard_ratio = math.exp(beta)
        lower = math.exp(beta - 1.959963984540054 * se)
        upper = math.exp(beta + 1.959963984540054 * se)
        p_value = float(fit.pvalues[0])
        return hazard_ratio, lower, upper, p_value
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def make_km_curve(
    time: np.ndarray,
    event: np.ndarray,
    cohort: str,
    panel: str,
    endpoint: str,
    group_label: str,
) -> pd.DataFrame:
    sf = km_object(time, event)
    rows = [{
        "cohort": cohort,
        "panel": panel,
        "endpoint": endpoint,
        "group": group_label,
        "time_months": 0.0,
        "survival_probability": 1.0,
        "ci_low": 1.0,
        "ci_high": 1.0,
        "n_at_risk": int(len(time)),
        "n_events_at_time": 0,
    }]
    if sf is None:
        return pd.DataFrame(rows)
    for t, s, se, n_risk, n_events in zip(sf.surv_times, sf.surv_prob, sf.surv_prob_se, sf.n_risk, sf.n_events):
        lower, upper = loglog_ci(float(s), float(se))
        rows.append({
            "cohort": cohort,
            "panel": panel,
            "endpoint": endpoint,
            "group": group_label,
            "time_months": float(t),
            "survival_probability": float(s),
            "ci_low": lower,
            "ci_high": upper,
            "n_at_risk": int(n_risk),
            "n_events_at_time": int(n_events),
        })
    return pd.DataFrame(rows)


def analyze_comparison(
    data: pd.DataFrame,
    cohort_label: str,
    panel_label: str,
    group_column: str,
    endpoint: str,
    time_column: str,
    event_column: str,
) -> tuple[dict[str, object], pd.DataFrame, list[dict[str, object]]]:
    use_columns = ["patient_id", group_column, time_column, event_column]
    for optional in ["age_years", "sex", "functional_mutation_count_total"]:
        if optional in data.columns:
            use_columns.append(optional)
    subset = data[use_columns].copy()
    subset[time_column] = pd.to_numeric(subset[time_column], errors="coerce")
    subset[event_column] = pd.to_numeric(subset[event_column], errors="coerce")
    subset = subset[
        subset[group_column].notna()
        & subset[time_column].notna()
        & subset[event_column].notna()
        & (subset[time_column] >= 0)
    ].copy()
    subset[group_column] = subset[group_column].astype(bool)
    subset[event_column] = subset[event_column].astype(int)

    mutated = subset[subset[group_column]]
    unmutated = subset[~subset[group_column]]
    mt = mutated[time_column].to_numpy(float)
    me = mutated[event_column].to_numpy(int)
    ut = unmutated[time_column].to_numpy(float)
    ue = unmutated[event_column].to_numpy(int)

    mutated_metrics = km_group_metrics(mt, me)
    unmutated_metrics = km_group_metrics(ut, ue)

    combined_time = subset[time_column].to_numpy(float)
    combined_event = subset[event_column].to_numpy(int)
    combined_group = subset[group_column].astype(int).to_numpy()
    try:
        logrank_chisq, logrank_p = survdiff(combined_time, combined_event, combined_group)
        logrank_chisq = float(logrank_chisq)
        logrank_p = float(logrank_p)
    except Exception:
        logrank_chisq, logrank_p = np.nan, np.nan

    hr, hr_low, hr_high, hr_p = cox_model(combined_time, combined_event, combined_group[:, None])

    age_hr = age_low = age_high = age_p = np.nan
    if "age_years" in subset.columns:
        age_subset = subset.dropna(subset=["age_years"]).copy()
        age_mutated_events = int(age_subset.loc[age_subset[group_column], event_column].sum())
        if len(age_subset) >= 30 and age_mutated_events >= 5:
            age = age_subset["age_years"].astype(float).to_numpy() / 10.0
            age = age - np.nanmean(age)
            exog = np.column_stack([age_subset[group_column].astype(int).to_numpy(), age])
            age_hr, age_low, age_high, age_p = cox_model(
                age_subset[time_column].to_numpy(float),
                age_subset[event_column].to_numpy(int),
                exog,
            )

    burden_hr = burden_low = burden_high = burden_p = np.nan
    if "functional_mutation_count_total" in subset.columns:
        burden_subset = subset.dropna(subset=["functional_mutation_count_total"]).copy()
        burden_mutated_events = int(burden_subset.loc[burden_subset[group_column], event_column].sum())
        if len(burden_subset) >= 30 and burden_mutated_events >= 5:
            burden = np.log1p(burden_subset["functional_mutation_count_total"].astype(float).to_numpy())
            burden = burden - np.nanmean(burden)
            exog = np.column_stack([burden_subset[group_column].astype(int).to_numpy(), burden])
            burden_hr, burden_low, burden_high, burden_p = cox_model(
                burden_subset[time_column].to_numpy(float),
                burden_subset[event_column].to_numpy(int),
                exog,
            )

    rmst_mutated = rmst(mt, me, 24.0)
    rmst_unmutated = rmst(ut, ue, 24.0)
    rmst_diff, rmst_low, rmst_high, rmst_p = bootstrap_rmst_difference(mt, me, ut, ue, tau=24.0)

    result = {
        "cohort": cohort_label,
        "panel": panel_label,
        "endpoint": endpoint,
        "n_with_endpoint": int(len(subset)),
        "n_mutated": mutated_metrics["n"],
        "n_unmutated": unmutated_metrics["n"],
        "events_mutated": mutated_metrics["events"],
        "events_unmutated": unmutated_metrics["events"],
        "median_months_mutated": mutated_metrics["median_months"],
        "median_ci_low_mutated": mutated_metrics["median_ci_low"],
        "median_ci_high_mutated": mutated_metrics["median_ci_high"],
        "median_months_unmutated": unmutated_metrics["median_months"],
        "median_ci_low_unmutated": unmutated_metrics["median_ci_low"],
        "median_ci_high_unmutated": unmutated_metrics["median_ci_high"],
        "survival_12m_mutated": mutated_metrics["survival_12m"],
        "survival_12m_unmutated": unmutated_metrics["survival_12m"],
        "survival_24m_mutated": mutated_metrics["survival_24m"],
        "survival_24m_unmutated": unmutated_metrics["survival_24m"],
        "logrank_chisq": logrank_chisq,
        "logrank_p_value": logrank_p,
        "cox_hr_mutated_vs_unmutated": hr,
        "cox_hr_ci_low": hr_low,
        "cox_hr_ci_high": hr_high,
        "cox_p_value": hr_p,
        "age_adjusted_cox_hr": age_hr,
        "age_adjusted_ci_low": age_low,
        "age_adjusted_ci_high": age_high,
        "age_adjusted_p_value": age_p,
        "mutation_burden_adjusted_cox_hr": burden_hr,
        "mutation_burden_adjusted_ci_low": burden_low,
        "mutation_burden_adjusted_ci_high": burden_high,
        "mutation_burden_adjusted_p_value": burden_p,
        "rmst_24m_mutated": rmst_mutated,
        "rmst_24m_unmutated": rmst_unmutated,
        "rmst_24m_difference_mutated_minus_unmutated": rmst_diff,
        "rmst_24m_difference_ci_low": rmst_low,
        "rmst_24m_difference_ci_high": rmst_high,
        "rmst_24m_bootstrap_p_value": rmst_p,
    }

    curves = pd.concat([
        make_km_curve(mt, me, cohort_label, panel_label, endpoint, "Splicing-panel mutated"),
        make_km_curve(ut, ue, cohort_label, panel_label, endpoint, "No splicing-panel mutation"),
    ], ignore_index=True)

    group_rows = []
    for label, metrics in [
        ("Splicing-panel mutated", mutated_metrics),
        ("No splicing-panel mutation", unmutated_metrics),
    ]:
        group_rows.append({
            "cohort": cohort_label,
            "panel": panel_label,
            "endpoint": endpoint,
            "group": label,
            **metrics,
        })
    return result, curves, group_rows


def baseline_rows(data: pd.DataFrame, cohort_label: str, panel_label: str, group_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_value, group_label in [(True, "Splicing-panel mutated"), (False, "No splicing-panel mutation")]:
        group = data[data[group_column].astype(bool) == group_value].copy()
        age = pd.to_numeric(group.get("age_years"), errors="coerce") if "age_years" in group.columns else pd.Series(dtype=float)
        burden = pd.to_numeric(group.get("functional_mutation_count_total"), errors="coerce") if "functional_mutation_count_total" in group.columns else pd.Series(dtype=float)
        sex = group.get("sex", pd.Series(dtype=str)).astype(str).str.upper()
        rows.append({
            "cohort": cohort_label,
            "panel": panel_label,
            "group": group_label,
            "n": int(len(group)),
            "age_n": int(age.notna().sum()),
            "age_median": float(age.median()) if age.notna().any() else np.nan,
            "age_q1": float(age.quantile(0.25)) if age.notna().any() else np.nan,
            "age_q3": float(age.quantile(0.75)) if age.notna().any() else np.nan,
            "female_n": int(sex.str.contains("FEMALE").sum()),
            "male_n": int(sex.str.contains("MALE").sum()),
            "mutation_burden_median": float(burden.median()) if burden.notna().any() else np.nan,
            "mutation_burden_q1": float(burden.quantile(0.25)) if burden.notna().any() else np.nan,
            "mutation_burden_q3": float(burden.quantile(0.75)) if burden.notna().any() else np.nan,
        })
    return rows


for key, url in FILES.items():
    download_file(url, WORK / f"{key}.txt")

clinical_patient = read_cbio_table(WORK / "clinical_patient.txt")
clinical_sample = read_cbio_table(WORK / "clinical_sample.txt")
mutations = pd.read_csv(WORK / "mutations.txt", sep="\t", comment="#", low_memory=False)

# Record raw clinical column names for transparent provenance.
pd.DataFrame({"patient_clinical_columns": clinical_patient.columns.tolist()}).to_csv(OUT / "clinical_patient_columns.csv", index=False)
pd.DataFrame({"sample_clinical_columns": clinical_sample.columns.tolist()}).to_csv(OUT / "clinical_sample_columns.csv", index=False)

patient_id_column = first_existing(clinical_patient.columns.tolist(), ["PATIENT_ID", "CASE_ID", "SUBMITTER_ID"])
if patient_id_column is None:
    raise RuntimeError("Unable to identify patient ID column in clinical patient file")
clinical_patient = clinical_patient.rename(columns={patient_id_column: "patient_id"})
clinical_patient["patient_id"] = clinical_patient["patient_id"].map(patient12)
clinical_patient = clinical_patient.drop_duplicates("patient_id")

sample_patient_column = first_existing(clinical_sample.columns.tolist(), ["PATIENT_ID", "CASE_ID"])
sample_id_column = first_existing(clinical_sample.columns.tolist(), ["SAMPLE_ID", "SAMPLE_NAME"])
if sample_patient_column is not None:
    clinical_sample = clinical_sample.rename(columns={sample_patient_column: "patient_id"})
    clinical_sample["patient_id"] = clinical_sample["patient_id"].map(patient12)
if sample_id_column is not None:
    clinical_sample = clinical_sample.rename(columns={sample_id_column: "sample_id"})
    clinical_sample["sample_id"] = clinical_sample["sample_id"].map(barcode15)

# Merge one primary-sample clinical row per patient to supplement patient-level fields.
if "patient_id" in clinical_sample.columns:
    if "sample_id" in clinical_sample.columns:
        primary_sample_clinical = clinical_sample[clinical_sample["sample_id"].map(is_primary_tumor)].copy()
        if primary_sample_clinical.empty:
            primary_sample_clinical = clinical_sample.copy()
    else:
        primary_sample_clinical = clinical_sample.copy()
    primary_sample_clinical = primary_sample_clinical.drop_duplicates("patient_id")
    add_columns = [column for column in primary_sample_clinical.columns if column not in {"patient_id", "sample_id"} and column not in clinical_patient.columns]
    if add_columns:
        clinical_patient = clinical_patient.merge(primary_sample_clinical[["patient_id"] + add_columns], on="patient_id", how="left")

# Standardize age and sex when available.
age_column = first_existing(clinical_patient.columns.tolist(), [
    "AGE", "AGE_AT_DIAGNOSIS", "AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS", "DIAGNOSIS_AGE",
])
sex_column = first_existing(clinical_patient.columns.tolist(), ["SEX", "GENDER"])
clinical_patient["age_years"] = normalize_age(clinical_patient[age_column]) if age_column else np.nan
clinical_patient["sex"] = clinical_patient[sex_column] if sex_column else np.nan

# Standardize available survival endpoints.
endpoint_candidates = [
    ("Overall survival", "OS_MONTHS", "OS_STATUS"),
    ("Progression-free survival", "PFS_MONTHS", "PFS_STATUS"),
    ("Disease-specific survival", "DSS_MONTHS", "DSS_STATUS"),
    ("Disease-free interval", "DFI_MONTHS", "DFI_STATUS"),
]
endpoints: list[tuple[str, str, str]] = []
for endpoint_label, time_candidate, status_candidate in endpoint_candidates:
    time_column = first_existing(clinical_patient.columns.tolist(), [time_candidate])
    status_column = first_existing(clinical_patient.columns.tolist(), [status_candidate])
    if time_column and status_column:
        standardized_time = endpoint_label.lower().replace("-", "_").replace(" ", "_") + "_months"
        standardized_event = endpoint_label.lower().replace("-", "_").replace(" ", "_") + "_event"
        clinical_patient[standardized_time] = numeric_series(clinical_patient[time_column])
        clinical_patient[standardized_event] = clinical_patient[status_column].map(parse_event)
        endpoints.append((endpoint_label, standardized_time, standardized_event))

if not endpoints:
    raise RuntimeError(f"No analyzable survival endpoints found. Columns: {clinical_patient.columns.tolist()}")

# Derive the all-sequenced primary IDH-wild-type genomic cohort.
if "Tumor_Sample_Barcode" not in mutations.columns:
    if "SAMPLE_ID" in mutations.columns:
        mutations["Tumor_Sample_Barcode"] = mutations["SAMPLE_ID"]
    else:
        raise RuntimeError("Mutation file lacks Tumor_Sample_Barcode and SAMPLE_ID")
mutations["sample_id"] = mutations["Tumor_Sample_Barcode"].map(barcode15)
mutations["patient_id"] = mutations["sample_id"].map(patient12)
mutations["Variant_Classification"] = mutations["Variant_Classification"].fillna("")
functional_mutations = mutations[mutations["Variant_Classification"].isin(FUNCTIONAL_VARIANT_CLASSES)].copy()

sequenced_samples = parse_case_list(WORK / "cases_sequenced.txt")
if not sequenced_samples:
    sequenced_samples = set(mutations["sample_id"].unique())
primary_samples = sorted(sample for sample in sequenced_samples if is_primary_tumor(sample))
primary_patients = sorted({patient12(sample) for sample in primary_samples})
representative_sample = {}
for sample in primary_samples:
    representative_sample.setdefault(patient12(sample), sample)

primary_functional = functional_mutations[functional_mutations["patient_id"].isin(primary_patients)].copy()
idh_mutated_patients = set(primary_functional.loc[primary_functional["Hugo_Symbol"].isin(IDH_GENES), "patient_id"])
idhwt_patients = sorted(set(primary_patients) - idh_mutated_patients)
idhwt_functional = primary_functional[primary_functional["patient_id"].isin(idhwt_patients)].copy()

all_groups = pd.DataFrame({"patient_id": idhwt_patients})
all_groups["sample_id"] = all_groups["patient_id"].map(representative_sample)
all_groups["focused_panel_mutated"] = all_groups["patient_id"].isin(
    set(idhwt_functional.loc[idhwt_functional["Hugo_Symbol"].isin(FOCUSED_GENES), "patient_id"])
)
all_groups["expanded_panel_mutated"] = all_groups["patient_id"].isin(
    set(idhwt_functional.loc[idhwt_functional["Hugo_Symbol"].isin(EXPANDED_GENES), "patient_id"])
)
all_groups["functional_mutation_count_total"] = all_groups["patient_id"].map(idhwt_functional.groupby("patient_id").size()).fillna(0).astype(int)
for panel_name, genes in [("focused", FOCUSED_GENES), ("expanded", EXPANDED_GENES)]:
    subset = idhwt_functional[idhwt_functional["Hugo_Symbol"].isin(genes)]
    mapping = subset.groupby("patient_id")["Hugo_Symbol"].agg(lambda values: ";".join(sorted(set(values))))
    all_groups[f"{panel_name}_mutated_genes"] = all_groups["patient_id"].map(mapping).fillna("")

matched_groups = pd.read_csv("data/matched_sample_groups.csv")
for column in ["focused_panel_mutated", "expanded_panel_mutated"]:
    matched_groups[column] = matched_groups[column].astype(str).str.upper().map({"TRUE": True, "FALSE": False})

all_data = all_groups.merge(clinical_patient, on="patient_id", how="left")
matched_data = matched_groups.merge(clinical_patient, on="patient_id", how="left")
all_data["in_expression_matched_cohort"] = all_data["patient_id"].isin(set(matched_groups["patient_id"]))

all_data.to_csv(OUT / "all_sequenced_idhwt_patient_level.csv", index=False)
matched_data.to_csv(OUT / "expression_matched_idhwt_patient_level.csv", index=False)

cohorts = [
    ("All sequenced primary IDH-wild-type TCGA-GBMs", all_data),
    ("Matched genomic-transcriptomic IDH-wild-type TCGA-GBMs", matched_data),
]
panels = [
    ("Focused exact panel (18 genes)", "focused_panel_mutated"),
    ("Expanded exact panel (25 genes)", "expanded_panel_mutated"),
]

comparison_rows: list[dict[str, object]] = []
curve_frames: list[pd.DataFrame] = []
group_metric_rows: list[dict[str, object]] = []
baseline_summary_rows: list[dict[str, object]] = []

for cohort_label, cohort_data in cohorts:
    for panel_label, group_column in panels:
        baseline_summary_rows.extend(baseline_rows(cohort_data, cohort_label, panel_label, group_column))
        for endpoint_label, time_column, event_column in endpoints:
            result, curves, group_rows = analyze_comparison(
                cohort_data,
                cohort_label,
                panel_label,
                group_column,
                endpoint_label,
                time_column,
                event_column,
            )
            comparison_rows.append(result)
            curve_frames.append(curves)
            group_metric_rows.extend(group_rows)

comparisons = pd.DataFrame(comparison_rows)
curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
group_metrics = pd.DataFrame(group_metric_rows)
baseline_summary = pd.DataFrame(baseline_summary_rows)

comparisons.to_csv(OUT / "survival_comparisons.csv", index=False)
curves.to_csv(OUT / "kaplan_meier_curves.csv", index=False)
group_metrics.to_csv(OUT / "survival_group_metrics.csv", index=False)
baseline_summary.to_csv(OUT / "baseline_characteristics.csv", index=False)

panel_definition = pd.DataFrame([
    {"panel": "Focused exact panel", "gene": gene, "gene_count": len(FOCUSED_GENES)} for gene in FOCUSED_GENES
] + [
    {"panel": "Expanded exact panel", "gene": gene, "gene_count": len(EXPANDED_GENES)} for gene in EXPANDED_GENES
])
panel_definition.to_csv(OUT / "survival_panel_definitions.csv", index=False)

summary = {
    "source_study": "cBioPortal DataHub: gbm_tcga_pan_can_atlas_2018",
    "clinical_patient_file": FILES["clinical_patient"],
    "clinical_sample_file": FILES["clinical_sample"],
    "mutation_file": FILES["mutations"],
    "all_primary_sequenced_patients": int(len(primary_patients)),
    "idh_mutated_patients_excluded": int(len(idh_mutated_patients)),
    "all_sequenced_idhwt_patients": int(len(all_groups)),
    "matched_idhwt_patients": int(len(matched_groups)),
    "focused_exact_gene_count": len(FOCUSED_GENES),
    "expanded_exact_gene_count": len(EXPANDED_GENES),
    "available_endpoints": [label for label, _, _ in endpoints],
    "comparisons": comparisons.replace({np.nan: None}).to_dict("records"),
}
(OUT / "survival_summary.json").write_text(json.dumps(summary, indent=2))

notes = [
    "TCGA-GBM splicing-related mutation survival analysis",
    "",
    "Primary clinical cohort: all primary, sequenced TCGA-GBM patients classified as IDH-wild type by absence of a functional IDH1/IDH2 mutation.",
    "Sensitivity cohort: the 144-patient genomic-transcriptomic cohort used for the preceding differential-expression analysis.",
    "Mutation-positive status required a protein-altering or canonical splice-site somatic variant; synonymous and noncoding calls were excluded.",
    "The exact prior focused list contains 18 genes and the expanded list 25 genes, despite earlier labels of 19 and 26.",
    "",
    "Methods: Kaplan-Meier estimates, two-sided log-rank test, univariable Cox proportional-hazards model, 12- and 24-month survival, and 24-month restricted mean survival time with stratified bootstrap confidence intervals.",
    "Age-adjusted and mutation-burden-adjusted Cox estimates are exploratory and are only reported when the mutation-positive group has at least five events.",
    "Because the mutation-positive groups are small, confidence intervals are expected to be wide and absence of statistical significance does not establish equivalence.",
]
(OUT / "survival_analysis_notes.txt").write_text("\n".join(notes))
print(json.dumps(summary, indent=2)[:12000])
