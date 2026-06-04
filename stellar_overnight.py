# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

DATA_DIR = r'C:\Users\abhishek\data\stellar'
SUB_DIR = r'C:\Users\abhishek\submissions\stellar'
os.makedirs(SUB_DIR, exist_ok=True)

print("Loading data...")
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

le_target = LabelEncoder()
y = le_target.fit_transform(train['class'])

def engineer(df):
    df = df.copy()
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    df['u_r'] = df['u'] - df['r']
    df['g_z'] = df['g'] - df['z']
    df['u_z'] = df['u'] - df['z']
    df['redshift_log'] = np.log1p(np.abs(df['redshift']))
    df['redshift_sq'] = df['redshift'] ** 2
    df['flux_total'] = df['u'] + df['g'] + df['r'] + df['i'] + df['z']
    df['flux_mean'] = df['flux_total'] / 5
    le = LabelEncoder()
    df['spectral_enc'] = le.fit_transform(df['spectral_type'].astype(str))
    df['galaxy_enc'] = le.fit_transform(df['galaxy_population'].astype(str))
    return df

cols = ['alpha','delta','u','g','r','i','z','redshift',
        'u_g','g_r','r_i','i_z','u_r','g_z','u_z',
        'redshift_log','redshift_sq','flux_total','flux_mean',
        'spectral_enc','galaxy_enc']

X = engineer(train)[cols]
X_test = engineer(test)[cols]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
best_score = 0
best_path = None

def run_experiment(name, model):
    global best_score, best_path
    print(f"\n[{name}] Training...")
    oof = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        model.fit(X.iloc[tr], y[tr])
        oof[val] = model.predict_proba(X.iloc[val])
        test_preds += model.predict_proba(X_test) / 10
    score = balanced_accuracy_score(y, oof.argmax(axis=1))
    print(f"[{name}] OOF: {score:.4f}")
    preds = le_target.inverse_transform(test_preds.argmax(axis=1))
    path = f'{SUB_DIR}\\{name}_{score:.4f}.csv'
    pd.DataFrame({'id': test['id'], 'class': preds}).to_csv(path, index=False)
    if score > best_score:
        best_score = score
        best_path = path
        print(f"[{name}] NEW BEST: {score:.4f}")
    return score, oof, test_preds

# Experiment 1: Strong LGBM
lgbm1 = lgb.LGBMClassifier(n_estimators=1000, max_depth=8, learning_rate=0.02,
    num_leaves=127, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbosity=-1, n_jobs=-1)
s1, oof1, tp1 = run_experiment("lgbm_v3", lgbm1)

# Experiment 2: XGBoost
xgb1 = xgb.XGBClassifier(n_estimators=800, max_depth=7, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    verbosity=0, eval_metric='mlogloss', n_jobs=-1)
s2, oof2, tp2 = run_experiment("xgb_v1", xgb1)

# Experiment 3: LGBM with different params
lgbm2 = lgb.LGBMClassifier(n_estimators=1200, max_depth=6, learning_rate=0.01,
    num_leaves=63, subsample=0.7, colsample_bytree=0.7,
    min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5,
    random_state=123, verbosity=-1, n_jobs=-1)
s3, oof3, tp3 = run_experiment("lgbm_v4", lgbm2)

# Experiment 4: Blend of all 3
print("\n[Blend] Combining all models...")
blend = (tp1 * 0.4 + tp2 * 0.3 + tp3 * 0.3)
blend_oof = (oof1 * 0.4 + oof2 * 0.3 + oof3 * 0.3)
blend_score = balanced_accuracy_score(y, blend_oof.argmax(axis=1))
print(f"[Blend] OOF: {blend_score:.4f}")
blend_preds = le_target.inverse_transform(blend.argmax(axis=1))
blend_path = f'{SUB_DIR}\\blend_{blend_score:.4f}.csv'
pd.DataFrame({'id': test['id'], 'class': blend_preds}).to_csv(blend_path, index=False)
if blend_score > best_score:
    best_score = blend_score
    best_path = blend_path
    print(f"[Blend] NEW BEST: {blend_score:.4f}")

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_score:.4f}")
print(f"BEST FILE: {best_path}")
print(f"\nSubmit with:")
print(f'$env:KAGGLE_API_TOKEN="KGAT_30fba9ceccf4cdf22431754d8c0072f7"')
print(f'kaggle competitions submit -c playground-series-s6e6 -f "{best_path}" -m "overnight best {best_score:.4f}"')
