import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
HORIZON_DAYS = 14                     # main horizon
SENSITIVITY_HORIZONS = [7, 14, 21, 28]
TARGET = 'long_resolution'            # 1 = resolution time > HORIZON_DAYS

PROJECT = os.environ.get('AGES_PROJECT',
                         '/content/drive/MyDrive/ages-resolution-time/pipeline')
RAW_CSV = os.environ.get('AGES_RAW_CSV', f'{PROJECT}/../Raw_AgES_Dataset.csv')

DIR_PREP       = f'{PROJECT}/01_data_prep'
DIR_BENCH_RND  = f'{PROJECT}/02_random_benchmark'
DIR_BENCH_CHR  = f'{PROJECT}/02_chronological_benchmark'
DIR_OPT_RND    = f'{PROJECT}/03_random_optimisation'
DIR_OPT_CHR    = f'{PROJECT}/03_chronological_optimisation'
DIR_PROT_A     = f'{PROJECT}/04_protocol_A'
DIR_PROT_B     = f'{PROJECT}/04_protocol_B'
DIR_PROT_C     = f'{PROJECT}/04_protocol_C'
DIR_STATS      = f'{PROJECT}/05_statistical_testing'
DIR_COMPARE    = f'{PROJECT}/06_protocol_comparison'
DIR_CROSS      = f'{PROJECT}/07_cross_project'
DIR_HORIZON    = f'{PROJECT}/08_horizon_sensitivity'
DIR_CATEGORY   = f'{PROJECT}/09_issue_category'

CLASS_LABELS = ['Short-resolution', 'Long-resolution']

MODEL_ORDER = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'AdaBoost',
               'Logistic Regression', 'MLP', 'Decision Tree']
MODEL_ABBR = {'Random Forest': 'RF', 'XGBoost': 'XGBoost', 'Gradient Boosting': 'GB',
              'AdaBoost': 'AdaBoost', 'Logistic Regression': 'LR', 'MLP': 'MLP',
              'Decision Tree': 'DT'}


def ensure(d):
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from scipy.stats import randint, uniform, loguniform


def pipe(clf, scale=False):
    steps = [('smote', SMOTE(random_state=RANDOM_STATE))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', clf))
    return ImbPipeline(steps)


def baseline_models():
    return {
        'Logistic Regression': pipe(LogisticRegression(max_iter=1000,
                                    random_state=RANDOM_STATE), scale=True),
        'Decision Tree': pipe(DecisionTreeClassifier(max_depth=5,
                              random_state=RANDOM_STATE)),
        'Random Forest': pipe(RandomForestClassifier(n_estimators=200, max_depth=8,
                              random_state=RANDOM_STATE, n_jobs=-1)),
        'Gradient Boosting': pipe(GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=RANDOM_STATE)),
        'AdaBoost': pipe(AdaBoostClassifier(n_estimators=100, learning_rate=0.1,
                         random_state=RANDOM_STATE)),
        'XGBoost': pipe(XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                        n_jobs=-1, eval_metric='logloss')),
        'MLP': pipe(MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                    max_iter=500, early_stopping=True, n_iter_no_change=10,
                    random_state=RANDOM_STATE), scale=True),
    }


N_ITER = 30


def search_spaces():
    return {
        'Random Forest': (pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)), {
            'clf__n_estimators': randint(100, 500), 'clf__max_depth': randint(4, 20),
            'clf__min_samples_split': randint(2, 20), 'clf__min_samples_leaf': randint(1, 10),
            'clf__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.8],
        }),
        'Gradient Boosting': (pipe(GradientBoostingClassifier(random_state=RANDOM_STATE)), {
            'clf__n_estimators': randint(100, 500), 'clf__max_depth': randint(2, 6),
            'clf__learning_rate': loguniform(0.01, 0.2), 'clf__subsample': uniform(0.6, 0.4),
            'clf__min_samples_leaf': randint(1, 20),
        }),
        'XGBoost': (pipe(XGBClassifier(random_state=RANDOM_STATE, n_jobs=1,
                         eval_metric='logloss')), {
            'clf__n_estimators': randint(100, 500), 'clf__max_depth': randint(3, 10),
            'clf__learning_rate': loguniform(0.01, 0.3), 'clf__subsample': uniform(0.5, 0.5),
            'clf__colsample_bytree': uniform(0.5, 0.5), 'clf__min_child_weight': randint(1, 10),
            'clf__gamma': uniform(0, 5),
        }),
        'MLP': (pipe(MLPClassifier(activation='relu', max_iter=300, early_stopping=True,
                     n_iter_no_change=10, random_state=RANDOM_STATE), scale=True), {
            'clf__hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
            'clf__alpha': loguniform(1e-5, 1e-1), 'clf__learning_rate_init': loguniform(1e-4, 1e-1),
            'clf__batch_size': [64, 128, 200],
        }),
        'Logistic Regression': (pipe(LogisticRegression(max_iter=2000,
                                random_state=RANDOM_STATE), scale=True), {
            'clf__C': loguniform(1e-3, 1e2), 'clf__penalty': ['l2'],
        }),
        'Decision Tree': (pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)), {
            'clf__max_depth': randint(3, 20), 'clf__min_samples_split': randint(2, 30),
            'clf__min_samples_leaf': randint(1, 20),
        }),
        'AdaBoost': (pipe(AdaBoostClassifier(random_state=RANDOM_STATE)), {
            'clf__n_estimators': randint(50, 400), 'clf__learning_rate': loguniform(0.01, 1.0),
        }),
    }


# --------------------------------------------------------------------------
# Splits and CV schemes
# --------------------------------------------------------------------------
def load_prepared():
    import joblib
    df = pd.read_csv(f'{DIR_PREP}/df_final.csv', parse_dates=['Created At'])
    feature_cols = joblib.load(f'{DIR_PREP}/feature_cols.pkl')
    return df, feature_cols


def random_split(df, feature_cols, target=TARGET):
    from sklearn.model_selection import train_test_split
    X = df[feature_cols].copy()
    y = df[target].copy()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def chronological_split(df, feature_cols, target=TARGET):
    d = df.sort_values('Created At').reset_index(drop=True)
    X = d[feature_cols].copy()
    y = d[target].copy()
    k = int(len(d) * 0.8)
    return (X.iloc[:k].copy(), X.iloc[k:].copy(), y.iloc[:k].copy(), y.iloc[k:].copy(), d, k)


def cv_random():
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)


def cv_chronological():
    from sklearn.model_selection import TimeSeriesSplit
    return TimeSeriesSplit(n_splits=10)


# --------------------------------------------------------------------------
# Threshold selection (F1 of the long-resolution class)
# --------------------------------------------------------------------------
from sklearn.base import clone
from sklearn.metrics import (precision_recall_curve, roc_auc_score, f1_score,
                             precision_score, recall_score, accuracy_score,
                             matthews_corrcoef, confusion_matrix)


def f1_max_threshold(y_true, proba):
    p, r, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-10)
    i = int(np.argmax(f1))
    return float(thr[i]), float(p[i]), float(r[i]), float(f1[i])


def oof_proba(model, X, y, cv):
    proba = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        proba[va] = m.predict_proba(X.iloc[va])[:, 1]
    return proba


def select_threshold_oof(model, X, y, cv, name='', verbose=True):
    proba = oof_proba(model, X, y, cv)
    ok = ~np.isnan(proba)
    thr, p, r, f1 = f1_max_threshold(np.asarray(y)[ok], proba[ok])
    if verbose:
        print(f'{name:22s} threshold={thr:.4f}  (OOF: P={p:.3f} R={r:.3f} F1={f1:.3f})')
    return thr


def select_threshold_on_test(model, X_test, y_test, name='', verbose=True):
    proba = model.predict_proba(X_test)[:, 1]
    thr, p, r, f1 = f1_max_threshold(y_test, proba)
    if verbose:
        print(f'{name:22s} threshold={thr:.4f} (chosen on TEST, Protocol A)  '
              f'P={p:.3f} R={r:.3f} F1={f1:.3f}')
    return thr


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metric_row(y_true, proba, pred):
    y_true = np.asarray(y_true)
    return {
        'Test AUC': roc_auc_score(y_true, proba),
        'Precision (Long)': precision_score(y_true, pred, pos_label=1, zero_division=0),
        'Recall (Long)': recall_score(y_true, pred, pos_label=1, zero_division=0),
        'F1 (Long)': f1_score(y_true, pred, pos_label=1, zero_division=0),
        'MCC': matthews_corrcoef(y_true, pred),
        'Accuracy': accuracy_score(y_true, pred),
    }


def rank_biserial(x, y):
    from scipy.stats import rankdata
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def save_versions(path):
    import sys, sklearn, xgboost, imblearn, scipy, statsmodels, matplotlib
    lines = [f'Python=={sys.version.split()[0]}']
    for lib in [pd, np, sklearn, xgboost, imblearn, scipy, statsmodels, matplotlib]:
        lines.append(f'{lib.__name__}=={lib.__version__}')
    open(path, 'w').write('\n'.join(lines) + '\n')
    return lines
