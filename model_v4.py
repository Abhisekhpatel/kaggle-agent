import os, numpy as np, pandas as pd, optuna, warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

DATA_DIR = r'C:\Users\abhishek\data'

def engineer_features(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['Age'] = df.groupby(['Title','Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FarePerPerson'] = df['FarePerPerson'].fillna(df['FarePerPerson'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['PclassSex'] = df['Pclass'].astype(str) + df['Sex'].astype(str)
    df['AgeClass'] = df['Age'] * df['Pclass']
    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['WomanOrChild'] = ((df['Sex'] == 'female') | (df['Age'] < 12)).astype(int)
    le = LabelEncoder()
    for col in ['Sex','Embarked','Title','PclassSex']:
        df[col] = le.fit_transform(df[col].astype(str))
    keep = ['Pclass','Sex','Age','Fare','FarePerPerson','FamilySize','IsAlone','Title','Embarked','PclassSex','AgeClass','IsChild','WomanOrChild']
    return df[keep]

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
ids = test['PassengerId'].copy()
y = train['Survived'].copy()
X = engineer_features(train)
X_test = engineer_features(test)

# Base models
models = {
    'lgbm': lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=-1),
    'xgb': xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, eval_metric='logloss'),
    'rf': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42),
    'lr': LogisticRegression(C=0.1, max_iter=1000, random_state=42)
}

# Generate OOF predictions for stacking
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof = np.zeros((len(X), len(models)))
test_preds = np.zeros((len(X_test), len(models)))

for i, (name, model) in enumerate(models.items()):
    print(f'Training {name}...')
    fold_test = np.zeros((len(X_test), 10))
    for j, (tr, val) in enumerate(skf.split(X, y)):
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[val, i] = model.predict_proba(X.iloc[val])[:, 1]
        fold_test[:, j] = model.predict_proba(X_test)[:, 1]
    test_preds[:, i] = fold_test.mean(axis=1)

# Meta learner
meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta.fit(oof, y)
meta_preds = meta.predict(test_preds)

# CV score on OOF
from sklearn.metrics import accuracy_score
oof_score = accuracy_score(y, (oof.mean(axis=1) > 0.5).astype(int))
meta_score = accuracy_score(y, meta.predict(oof))
print(f'OOF mean: {oof_score:.4f}')
print(f'Meta CV: {meta_score:.4f}')

# Validation gate
MIN_SCORE = 0.78
if meta_score < MIN_SCORE:
    print(f'GATE FAILED: {meta_score:.4f} < {MIN_SCORE}. Not submitting.')
else:
    os.makedirs(r'C:\Users\abhishek\submissions', exist_ok=True)
    sub_path = rf'C:\Users\abhishek\submissions\submission_v4_stacking_{meta_score:.4f}.csv'
    pd.DataFrame({'PassengerId': ids, 'Survived': meta_preds.astype(int)}).to_csv(sub_path, index=False)
    print(f'GATE PASSED. Saved to {sub_path}')
