#pip3 install openml pandas numpy groq scikit-learn umap-learn matplotlib

import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
import os
warnings.filterwarnings('ignore')

OPENML_DATASET_ID = 40701
RANDOM_STATE = 42
TEST_SIZE = 0.2

def ks_test_numeric(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict:
    results = {'ks_tests': {}, 'mean_diff': {}, 'std_diff': {}}
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    print(numeric_cols)

    for col in numeric_cols:
        if col in synth_df.columns:
            real_vals = real_df[col].dropna()
            synth_vals = synth_df[col].dropna()
            if len(real_vals) > 0 and len(synth_vals) > 0:
                ks_stat, p_value = stats.ks_2samp(real_vals, synth_vals)
                results['ks_tests'][col] = {'statistic': ks_stat, 'p_value': p_value}
                results['mean_diff'][col] = abs(real_vals.mean() - synth_vals.mean())
                results['std_diff'][col] = abs(real_vals.std() - synth_vals.std())

    return results


def chi2_test_categorical(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict:
    results = {'chi2_tests': {}}
    categorical_cols = real_df.select_dtypes(exclude=[np.number]).columns.tolist()
    print(categorical_cols)

    for col in categorical_cols:
        if col in synth_df.columns:
            real_counts = real_df[col].value_counts()
            synth_counts = synth_df[col].value_counts()
            all_categories = set(real_counts.index) | set(synth_counts.index)
            real_freq = [real_counts.get(cat, 0) for cat in all_categories]
            synth_freq = [synth_counts.get(cat, 0) for cat in all_categories]

            if sum(real_freq) > 0 and sum(synth_freq) > 0:
                scale_factor = sum(synth_freq) / sum(real_freq)
                expected_freq = [x * scale_factor for x in real_freq]
                chi2_stat, p_value = stats.chisquare(f_obs=synth_freq, f_exp=expected_freq)
                results['chi2_tests'][col] = {'statistic': chi2_stat, 'p_value': p_value}

    return results


def correlation_difference(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) <= 1:
        return None

    real_corr = real_df[numeric_cols].corr()
    synth_corr = synth_df[numeric_cols].corr()
    corr_diff = np.abs(real_corr - synth_corr).mean().mean()

    return corr_diff


def calculate_statistical_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict:
    results = {}

    numeric_results = ks_test_numeric(real_df, synth_df)
    results.update(numeric_results)

    chi2_results = chi2_test_categorical(real_df, synth_df)
    results.update(chi2_results)

    results['correlation_diff'] = correlation_difference(real_df, synth_df)

    return results

def load_churn_data_with_target():
    dataset = openml.datasets.get_dataset(OPENML_DATASET_ID)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )

    print(y.value_counts())

    return X, y, dataset.default_target_attribute


def calculate_metrics(y_true, y_pred, y_prob=None) -> Dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc_roc'] = None
    else:
        metrics['auc_roc'] = None

    return metrics

def print_metrics(metrics: Dict):
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    if metrics['auc_roc']:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

def plot_comparison_results(ml_results: Dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Synthetic Data Quality Evaluation', fontsize=16, fontweight='bold')

    models = list(ml_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        train_real_test_real = [ml_results[m]['train_real_test_real'][metric] for m in models]
        train_real_test_synth = [ml_results[m]['train_real_test_synth'][metric] for m in models]
        train_synth_test_real = [ml_results[m]['train_synth_test_real'][metric] for m in models]
        train_both_test_real = [ml_results[m]['train_both_test_real'][metric] for m in models]

        x = np.arange(len(models))
        width = 0.2

        ax.bar(x - 1.5*width, train_real_test_real, width, label='Train Real / Test Real', alpha=0.8, color='#2E86AB')
        ax.bar(x - 0.5*width, train_real_test_synth, width, label='Train Real / Test Synth', alpha=0.8, color='#A23B72')
        ax.bar(x + 0.5*width, train_synth_test_real, width, label='Train Synth / Test Real', alpha=0.8, color='#F18F01')
        ax.bar(x + 1.5*width, train_both_test_real, width, label='Train Both / Test Real', alpha=0.8, color='#06A77D')

        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

def plot_utility_scores(ml_results: Dict):
    models = list(ml_results.keys())
    utility_scores = [ml_results[m]['utility_score'] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#28a745' if score > 0.95 else '#ffc107' if score > 0.90 else '#dc3545'
              for score in utility_scores]

    bars = ax.barh(models, utility_scores, color=colors, alpha=0.8, edgecolor='black')

    ax.axvline(x=0.95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (>95%)')
    ax.axvline(x=0.90, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good (>90%)')
    ax.axvline(x=0.85, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Acceptable (>85%)')

    for i, (bar, score) in enumerate(zip(bars, utility_scores)):
        ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontweight='bold')

    ax.set_xlabel('Utility Score (Train Synth Test Real / Train Real Test Real)', fontsize=12, fontweight='bold')
    ax.set_title('ML Utility Score by Model', fontsize=14, fontweight='bold')
    ax.set_xlim([0.7, 1.05])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

def identify_column_types(X: pd.DataFrame) -> Tuple[list, list]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols
        ))
    preprocessor = ColumnTransformer(transformers)
    return preprocessor


def encode_target(y: pd.Series) -> Tuple[pd.Series]:
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return y_encoded, le
    else:
        return y.values, None


def prepare_data_for_ml(X: pd.DataFrame, y: pd.Series) -> Tuple[ColumnTransformer, pd.Series]:
    numeric_cols, categorical_cols = identify_column_types(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    y_encoded, le = encode_target(y)
    return preprocessor, y_encoded, le

def align_synthetic_data(X_real_train: pd.DataFrame, X_synth: pd.DataFrame) -> pd.DataFrame:
    X_synth_aligned = X_synth[X_real_train.columns].copy()
    for col in X_real_train.columns:
        X_synth_aligned[col] = X_synth_aligned[col].astype(X_real_train[col].dtype)
    return X_synth_aligned

def encode_targets(y_real_train, y_real_test, y_synth):
    y_train_str = y_real_train.astype(str)
    y_test_str = y_real_test.astype(str)
    y_synth_str = y_synth.astype(str)

    all_labels = pd.concat([y_train_str, y_test_str, y_synth_str], axis=0)
    le = LabelEncoder().fit(all_labels)

    return le.transform(y_train_str), le.transform(y_test_str), le.transform(y_synth_str), le

def safe_predict_proba(model, X):
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        return y_prob
    return None

def evaluate_model_scenarios(model, X_train_real, y_train_real, X_test_real, y_test_real,
                             X_synth, y_synth_enc) -> Dict:
    results = {}

    print("\n1. Train on Real, Test on Real:")
    model_real = model.__class__(**model.get_params())
    model_real.fit(X_train_real, y_train_real)
    y_pred = model_real.predict(X_test_real)
    y_prob = safe_predict_proba(model_real, X_test_real)
    results['train_real_test_real'] = calculate_metrics(y_test_real, y_pred, y_prob)
    print_metrics(results['train_real_test_real'])

    print("\n2. Train on Real, Test on Synthetic:")
    y_pred = model_real.predict(X_synth)
    y_prob = safe_predict_proba(model_real, X_synth)
    results['train_real_test_synth'] = calculate_metrics(y_synth_enc, y_pred, y_prob)
    print_metrics(results['train_real_test_synth'])

    print("\n3. Train on Synthetic, Test on Real:")
    model_synth = model.__class__(**model.get_params())
    model_synth.fit(X_synth, y_synth_enc)
    y_pred = model_synth.predict(X_test_real)
    y_prob = safe_predict_proba(model_synth, X_test_real)
    results['train_synth_test_real'] = calculate_metrics(y_test_real, y_pred, y_prob)
    print_metrics(results['train_synth_test_real'])

    print("\n4. Train on Synthetic + Real, Test on Real:")
    X_combined = np.vstack([X_train_real, X_synth])
    y_combined = np.hstack([y_train_real, y_synth_enc])
    model_combined = model.__class__(**model.get_params())
    model_combined.fit(X_combined, y_combined)
    y_pred = model_combined.predict(X_test_real)
    y_prob = safe_predict_proba(model_combined, X_test_real)
    results['train_both_test_real'] = calculate_metrics(y_test_real, y_pred, y_prob)
    print_metrics(results['train_both_test_real'])

    utility_score = results['train_synth_test_real']['accuracy'] / results['train_real_test_real']['accuracy']
    results['utility_score'] = utility_score
    status = ("EXCELLENT" if utility_score > 0.95 else
              "GOOD" if utility_score > 0.90 else
              "ACCEPTABLE" if utility_score > 0.85 else
              "POOR")
    print(f"\nUtility Score: {utility_score:.4f} ({utility_score*100:.2f}% of baseline) | Status: {status}")

    return results

def evaluate_ml_utility(X_real: pd.DataFrame, y_real: pd.Series,
                        X_synth: pd.DataFrame, y_synth: pd.Series) -> Dict:

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_real
    )
    print(f"\nData splits:")
    print(f"  Real train: {X_real_train.shape}")
    print(f"  Real test: {X_real_test.shape}")
    print(f"  Synthetic: {X_synth.shape}")

    X_synth_proc = align_synthetic_data(X_real_train, X_synth)

    preprocessor, _, _ = prepare_data_for_ml(X_real_train, y_real_train)
    X_real_train_proc = preprocessor.fit_transform(X_real_train)
    X_real_test_proc = preprocessor.transform(X_real_test)
    X_synth_proc = preprocessor.transform(X_synth_proc)

    y_real_train_enc, y_real_test_enc, y_synth_enc, _ = encode_targets(
        y_real_train, y_real_test, y_synth
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    results = {}
    for model_name, model in models.items():
        print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
        results[model_name] = evaluate_model_scenarios(
            model,
            X_real_train_proc,
            y_real_train_enc,
            X_real_test_proc,
            y_real_test_enc,
            X_synth_proc,
            y_synth_enc
        )

    return results

def evaluate_synthetic_data(X_real: pd.DataFrame, y_real: pd.Series,
                           X_synth: pd.DataFrame, y_synth: pd.Series):

    fidelity_results = calculate_statistical_fidelity(X_real, X_synth)
    ml_results = evaluate_ml_utility(X_real, y_real, X_synth, y_synth)
    plot_comparison_results(ml_results)
    plot_utility_scores(ml_results)

    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)

    print("\n✓ Statistical Fidelity:")
    if fidelity_results['ks_tests']:
        ks_passes = sum(1 for v in fidelity_results['ks_tests'].values() if v['p_value'] > 0.05)
        ks_total = len(fidelity_results['ks_tests'])
        print(f"  KS Test Pass Rate: {ks_passes}/{ks_total} ({100*ks_passes/ks_total:.1f}%)")

    if fidelity_results['correlation_diff']:
        print(f"  Correlation Difference: {fidelity_results['correlation_diff']:.4f}")

    print("\nML Utility Scores:")
    for model_name, results in ml_results.items():
        print(f"  {model_name:25s}: {results['utility_score']:.4f}")

    avg_utility = np.mean([r['utility_score'] for r in ml_results.values()])
    print(f"\nAverage Utility Score: {avg_utility:.4f} ({avg_utility*100:.2f}%)")

    if avg_utility > 0.95:
        print("VERDICT: EXCELLENT - Synthetic data is highly representative")
    elif avg_utility > 0.90:
        print("VERDICT: GOOD - Synthetic data is suitable for most use cases")
    elif avg_utility > 0.85:
        print("VERDICT: ACCEPTABLE - Synthetic data may need improvement")
    else:
        print("VERDICT: POOR - Synthetic data needs significant improvement")

    return fidelity_results, ml_results

# from google.colab import drive
# drive.mount('/content/drive')
# !ls -lh /content/drive/MyDrive

if __name__ == "__main__":
    X_real, y_real, target_name = load_churn_data_with_target()
    X_real = X_real[0:500]
    y_real = y_real[0:500]
    print(f"Target variable name: '{target_name}'")

    data_dir = 'data'
    synth_file_path = os.path.join(data_dir, 'df_synthetic.csv')
    df_synth = pd.read_csv(synth_file_path)

    X_synth = df_synth.drop(columns=[target_name])
    y_synth = df_synth[target_name]

    fidelity_results, ml_results = evaluate_synthetic_data(X_real, y_real, X_synth, y_synth)