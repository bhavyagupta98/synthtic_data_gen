# pip3 install openml pandas numpy groq scikit-learn umap-learn matplotlib

import openml
import pandas as pd
import numpy as np
import json
import time
import os
from groq import Groq
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import hashlib

GROQ_API_KEY = ""
client = Groq(api_key=GROQ_API_KEY)

OPENML_DATASET_ID = 40701
SUBSAMPLE_N = 500
SYNTH_N = 1000
BATCH_SIZE = 50
EXAMPLE_SIZE = 5
MODEL_NAME = "openai/gpt-oss-20b"
TEMPERATURE = 0.2
MAX_TOKENS = 50000
USE_COLUMN_ENCODING = True

def load_openml_dataset(dataset_id):
    ds = openml.datasets.get_dataset(dataset_id)
    X, _, _, _ = ds.get_data(dataset_format="dataframe", target=None)
    return X


def identify_column_types(df: pd.DataFrame) -> Dict[str, str]:
    type_map = {}
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            type_map[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(df[col]):
            type_map[col] = "number"
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                type_map[col] = "categorical"
            else:
                type_map[col] = "text"
        else:
            type_map[col] = "string"
    return type_map


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    type_map = identify_column_types(df)
    return [col for col, dtype in type_map.items()
            if dtype in ["categorical", "text", "string", "boolean"]]


def get_column_stats(df: pd.DataFrame, type_map: Dict[str, str]) -> Dict[str, Dict]:
    stats = {}
    for col in df.columns:
        col_stats = {}
        col_type = type_map[col]

        if col_type == "number":
            col_stats["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            col_stats["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
            col_stats["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
            col_stats["median"] = float(df[col].median()) if not pd.isna(df[col].median()) else None
        elif col_type in ["categorical", "boolean"]:
            value_counts = df[col].value_counts().head(10)
            col_stats["top_values"] = value_counts.index.tolist()
            col_stats["frequencies"] = value_counts.values.tolist()
        elif col_type == "text":
            col_stats["sample_values"] = df[col].dropna().head(3).tolist()

        col_stats["null_count"] = int(df[col].isna().sum())
        stats[col] = col_stats

    return stats


def create_column_mapping(columns: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    col_to_code = {col: f"c{i}" for i, col in enumerate(columns)}
    code_to_col = {v: k for k, v in col_to_code.items()}
    return col_to_code, code_to_col


def encode_examples(examples: List[Dict], col_to_code: Dict[str, str]) -> List[Dict]:
    return [{col_to_code[k]: v for k, v in row.items()} for row in examples]


def encode_stats(stats: Dict[str, Dict], col_to_code: Dict[str, str]) -> Dict[str, Dict]:
    return {col_to_code[k]: v for k, v in stats.items()}


def decode_synthetic(synth_rows: List[Dict], code_to_col: Dict[str, str]) -> List[Dict]:
    return [{code_to_col.get(k, k): v for k, v in row.items()} for row in synth_rows]


def build_prompt(columns: List[str], full_df: pd.DataFrame,
                 n_return: int, example_size: int = EXAMPLE_SIZE,
                 use_encoding: bool = False) -> str:
    type_map = identify_column_types(full_df)

    example_rows = full_df.sample(min(example_size, len(full_df)), random_state=None)
    examples_json = example_rows.to_dict(orient="records")

    print(examples_json)

    stats = get_column_stats(full_df, type_map)

    if use_encoding:
        col_to_code, _ = create_column_mapping(columns)

        examples_json = encode_examples(examples_json, col_to_code)
        stats_encoded = encode_stats(stats, col_to_code)

        mapping_desc = json.dumps(col_to_code, indent=2)
        hint_lines = "; ".join([f"{col_to_code[k]}: {v}" for k, v in type_map.items()])

        prompt = f"""
You are generating synthetic tabular data. Column names are encoded to save tokens.

Column mapping (code -> full name):
{mapping_desc}

Column types: {hint_lines}

Column statistics (from full dataset):
{json.dumps(stats_encoded, indent=2)}

Here are {len(examples_json)} example rows (using encoded column names):
{json.dumps(examples_json, indent=2)}

Generate exactly {n_return} NEW rows following the same patterns and distributions.
IMPORTANT:
- Use the ENCODED column names (c0, c1, etc.)
- Keep numeric columns as numbers (no quotes)
- Follow the statistical distributions shown above
- Keep categorical/text as strings
- Create DIVERSE rows, avoid repetition
- Output ONLY valid JSON array of objects

Return only JSON (no explanation).
"""
    else:
        col_list = ", ".join(columns)
        hint_lines = "; ".join([f"{k}: {v}" for k, v in type_map.items()])

        prompt = f"""
You are given a tabular dataset with columns: {col_list}.
Column types: {hint_lines}.

Column statistics (from full dataset):
{json.dumps(stats, indent=2)}

Here are {len(examples_json)} example rows:
{json.dumps(examples_json, indent=2)}

Generate exactly {n_return} NEW rows that follow the same schema and distributions.
IMPORTANT:
- Keep numeric columns as numbers (no quotes)
- Follow the statistical distributions shown above
- Keep categorical/text columns as strings
- Create DIVERSE rows, avoid duplicates
- Output ONLY valid JSON array of objects

Return only JSON (no explanation).
"""

    return prompt.strip()


def estimate_prompt_tokens(prompt: str) -> int:
    return len(prompt) // 4


def generate_synthetic_rows(prompt: str, model: str = MODEL_NAME,
                           temperature: float = TEMPERATURE,
                           max_tokens: int = MAX_TOKENS,
                           retries: int = 3) -> List[Dict]:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            text = response.choices[0].message.content.strip()
            print(text[0])
            data = json.loads(text)
            if isinstance(data, str):
                data = json.loads(data)

            return data
        except Exception as e:
            print(f"  ⚠ Attempt {attempt+1} failed: {e}")
            time.sleep(1 + attempt * 2)

    raise RuntimeError("Failed to generate or parse JSON after retries.")


def row_hash(row: Dict) -> str:
    sorted_items = sorted(row.items())
    row_str = json.dumps(sorted_items, sort_keys=True, default=str)
    return hashlib.md5(row_str.encode()).hexdigest()


def generate_synthetic_batched(df_sample: pd.DataFrame,
                               total_samples: int,
                               batch_size: int = BATCH_SIZE,
                               example_size: int = EXAMPLE_SIZE,
                               use_encoding: bool = USE_COLUMN_ENCODING) -> pd.DataFrame:
    columns = df_sample.columns.tolist()
    all_synth_rows = []
    seen_hashes = set()

    for _, row in df_sample.iterrows():
        seen_hashes.add(row_hash(row.to_dict()))

    n_batches = (total_samples + batch_size - 1) // batch_size
    samples_generated = 0

    print(f"\n{'='*60}")
    print(f"Generating {total_samples} samples in {n_batches} batches")
    print(f"Batch size: {batch_size} | Example size: {example_size}")
    print(f"Column encoding: {use_encoding}")
    print(f"{'='*60}\n")

    for batch_idx in range(n_batches):
        samples_needed = min(batch_size, total_samples - samples_generated)

        if samples_needed <= 0:
            break

        print(f"📦 Batch {batch_idx + 1}/{n_batches}: Requesting {samples_needed} samples...")

        prompt = build_prompt(columns, df_sample, samples_needed, example_size, use_encoding)

        estimated_tokens = estimate_prompt_tokens(prompt)
        print(f"   Estimated prompt tokens: ~{estimated_tokens}")

        try:
            synth_records = generate_synthetic_rows(prompt)

            if use_encoding:
                _, code_to_col = create_column_mapping(columns)
                synth_records = decode_synthetic(synth_records, code_to_col)

            unique_records = []
            for record in synth_records:
                h = row_hash(record)
                if h not in seen_hashes:
                    unique_records.append(record)
                    seen_hashes.add(h)

            all_synth_rows.extend(unique_records)
            prev_generated = samples_generated
            samples_generated += len(unique_records)

            duplicates_found = len(synth_records) - len(unique_records)
            if duplicates_found > 0:
                print(f"   ⚠ Filtered {duplicates_found} duplicate(s)")
            print(f"   ✓ Generated {len(unique_records)} unique samples")
            print(f"   📊 Progress: {samples_generated}/{total_samples} ({100*samples_generated/total_samples:.1f}%)\n")

            time.sleep(0.5)

        except Exception as e:
            print(f"   ✗ Batch {batch_idx + 1} failed: {e}\n")
            continue

    print(f"{'='*60}")
    print(f"✓ Generation complete: {samples_generated} unique samples generated")
    print(f"{'='*60}\n")

    df_synth = coerce_synthetic_df(all_synth_rows, df_sample)
    return df_synth


def coerce_synthetic_df(synth_records: List[Dict], reference_df: pd.DataFrame) -> pd.DataFrame:
    if not synth_records:
        print("⚠ Warning: No synthetic records generated!")
        return pd.DataFrame(columns=reference_df.columns)

    df_s = pd.DataFrame(synth_records)

    for c in reference_df.columns:
        if c not in df_s.columns:
            df_s[c] = np.nan

    df_s = df_s[reference_df.columns.tolist()].copy()

    for c in reference_df.columns:
        if pd.api.types.is_numeric_dtype(reference_df[c]):
            df_s[c] = pd.to_numeric(df_s[c], errors="coerce")
        else:
            df_s[c] = df_s[c].astype(object).where(df_s[c].notnull(), None)

    return df_s


def encode_for_embedding(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df_all = pd.concat([real_df, synth_df], ignore_index=True)

    numeric_cols = [c for c in df_all.columns
                   if pd.api.types.is_numeric_dtype(real_df[c])]
    categorical_cols = get_categorical_columns(real_df)

    print(f"Encoding for visualization:")
    print(f"  - Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
    print(f"  - Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=False,
                                                  dtype=float),
                            categorical_cols))

    ct = ColumnTransformer(transformers)
    X_enc = ct.fit_transform(df_all)

    n_real = len(real_df)
    return X_enc[:n_real], X_enc[n_real:]


def plot_embeddings(X_real: np.ndarray, X_synth: np.ndarray, title_prefix: str = ""):
    labels = np.array(["real"] * X_real.shape[0] + ["synth"] * X_synth.shape[0])

    print(f"\nGenerating embeddings...")

    print("  - Computing PCA...")
    pca = PCA(n_components=2)
    Z_real_pca = pca.fit_transform(X_real)
    Z_synth_pca = pca.transform(X_synth)
    Z_pca = np.vstack([Z_real_pca, Z_synth_pca])

    print("  - Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    Z_real_umap = reducer.fit_transform(X_real)
    Z_synth_umap = reducer.transform(X_synth)
    Z_umap = np.vstack([Z_real_umap, Z_synth_umap])

    print("  - Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"real": "#2E86AB", "synth": "#A23B72"}

    for lbl, marker, alpha in [("real", "o", 0.6), ("synth", "^", 0.8)]:
        mask = labels == lbl
        axes[0].scatter(Z_pca[mask, 0], Z_pca[mask, 1],
                       label=lbl, alpha=alpha, s=40, marker=marker,
                       color=colors[lbl], edgecolors='white', linewidth=0.5)
        axes[1].scatter(Z_umap[mask, 0], Z_umap[mask, 1],
                       label=lbl, alpha=alpha, s=40, marker=marker,
                       color=colors[lbl], edgecolors='white', linewidth=0.5)

    axes[0].set_title(f"{title_prefix} PCA Projection", fontsize=14, fontweight='bold')
    axes[1].set_title(f"{title_prefix} UMAP Projection", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")

    for ax in axes:
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.show()
    print("  ✓ Plot complete!\n")

def main():

    print(f"\nLoading OpenML dataset {OPENML_DATASET_ID}")
    df = load_openml_dataset(OPENML_DATASET_ID)
    print(f"    Dataset shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")


    print(f"\nSampling {SUBSAMPLE_N} rows for reference")
    df_sample = df.sample(SUBSAMPLE_N, random_state=42)

    print(df_sample.shape)


    type_map = identify_column_types(df_sample)


    print(f"\nGenerating {SYNTH_N} synthetic samples")

    df_synth = generate_synthetic_batched(
        df_sample,
        total_samples=SYNTH_N,
        batch_size=BATCH_SIZE,
        example_size=EXAMPLE_SIZE,
        use_encoding=USE_COLUMN_ENCODING
    )


    print(f"Synthetic data summary:")
    print(f"    Shape: {df_synth.shape}")
    print(f"    Memory usage: {df_synth.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\n    First few rows:")
    print(df_synth.head())


    print(f"\nEncoding data for visualization")
    X_real, X_synth = encode_for_embedding(df_sample, df_synth)
    print(f"    Encoded shape: {X_real.shape}")


    print(f"\nGenerating embeddings plot")
    plot_embeddings(X_real, X_synth, title_prefix="Dataset")

    print("="*60)
    print("Process complete!")
    print("="*60 + "\n")

    return df_sample, df_synth

if __name__ == "__main__":
    df_sample, df_synth = main()

    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'df_synthetic.csv')
    df_synth.to_csv(output_path, index=False)
    print(f"\n✓ Synthetic data saved to: {output_path}")