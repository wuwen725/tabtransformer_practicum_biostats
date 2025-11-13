# src/data_utils.py

from pathlib import Path
import pandas as pd


# =========================
# Parsing & Integration
# =========================

def parse_series_matrix(file_path: str) -> pd.DataFrame:
    """
    Parse a GEO series matrix file to extract sample IDs and clinical characteristics.
    (Adapted from feature_description GSE164641 code.)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Get sample IDs (GSMs)
    sample_ids = []
    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = line.strip().split('\t')[1:]
            sample_ids = [s.replace('"', '') for s in sample_ids]
            break

    characteristics = {}
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1"):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            first_value_cell = parts[1].replace('"', '')
            if ':' not in first_value_cell:
                continue

            feature_name_raw = first_value_cell.split(':')[0]
            feature_name = feature_name_raw.strip().replace(' ', '_').lower()

            prefix_to_remove = f"{feature_name_raw}: "
            values = [v.replace('"', '') for v in parts[1:]]
            cleaned_values = [v.replace(prefix_to_remove, "").strip() for v in values]

            characteristics[feature_name] = cleaned_values

    clinical_df = pd.DataFrame(characteristics)
    clinical_df.insert(0, 'sample_id', sample_ids)
    return clinical_df


def load_gene_id_map(annot_file_path: str) -> dict:
    """
    Load gene ID -> symbol mapping from Human.GRCh38.p13.annot.tsv.
    """
    annot_df = pd.read_csv(
        annot_file_path,
        sep="\t",
        header=None,
        usecols=[0, 1],
        on_bad_lines="skip",
    )
    annot_df.columns = ["gene_id", "gene_symbol"]
    annot_df.dropna(subset=["gene_id", "gene_symbol"], inplace=True)
    return dict(zip(annot_df["gene_id"].astype(str), annot_df["gene_symbol"]))


def build_master_dataframe(
    series_matrix_path: str,
    raw_counts_path: str,
    annot_path: str,
    out_csv_path: str = "GSE164641_master_dataframe.csv",
) -> pd.DataFrame:
    """
    Reproduce the 'Part 1' integration pipeline:
    - parse clinical info
    - load raw counts
    - map Ensembl IDs to gene symbols
    - merge into a single master dataframe
    """
    clinical_df = parse_series_matrix(series_matrix_path)

    # Dataset-specific renamings (for GSE164641)
    if "life_time_risk_(tyrer-cuzick_score)" in clinical_df.columns:
        clinical_df = clinical_df.rename(
            columns={"life_time_risk_(tyrer-cuzick_score)": "tyrer_cuzick_score"}
        )
    if "risk_category" in clinical_df.columns:
        clinical_df = clinical_df.rename(columns={"risk_category": "target"})

    raw_counts_df = pd.read_csv(raw_counts_path, sep="\t", index_col=0)
    raw_counts_df = raw_counts_df.T
    raw_counts_df.reset_index(inplace=True)
    raw_counts_df = raw_counts_df.rename(columns={"index": "sample_id"})

    raw_counts_df.columns = raw_counts_df.columns.map(str)
    gene_id_map = load_gene_id_map(annot_path)
    if gene_id_map:
        raw_counts_df = raw_counts_df.rename(columns=gene_id_map)

    master_df = pd.merge(clinical_df, raw_counts_df, on="sample_id", how="inner")
    master_df = master_df.set_index("sample_id")
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(out_csv_path)
    return master_df


# =========================
# Convenience helpers
# =========================

def load_master_dataframe(path: str) -> pd.DataFrame:
    """Load master dataframe (clinical + expression)."""
    return pd.read_csv(path, index_col=0)


def split_features_target(
    master_df: pd.DataFrame,
    target_col: str = "target",
    mapping: dict | None = None,
):
    """
    Split master_df into (X, y).
    Default mapping is {'High':1, 'Average':0} for GSE164641.
    """
    if mapping is None:
        mapping = {"High": 1, "Average": 0}

    y = master_df[target_col].map(mapping)
    X = master_df.drop(columns=[target_col])
    return X, y
