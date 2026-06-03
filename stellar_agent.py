import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb

DATA_DIR = r'C:\Users\abhishek\data\stellar'

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

def engineer(df):
    df = df.copy()
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    df['redshift_log'] = np.log1p(np.abs(df['redshift']))
    return df

cols = ['alpha','delta','u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','redshift_log']
X = engineer(train)[cols]
le = LabelEncoder()
y = le.fit_transform(train['class'])
X_test = engineer(test)[cols]

model = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=-1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))

for fold, (tr, val) in enumerate(cv.split(X, y)):
    print(f'Fold {fold+1}/5...')
    model.fit(X.iloc[tr], y[tr])
    oof[val] = model.predict_proba(X.iloc[val])
    test_preds += model.predict_proba(X_test) / 5

score = balanced_accuracy_score(y, oof.argmax(axis=1))
print(f'OOF: {score:.4f}')

preds = le.inverse_transform(test_preds.argmax(axis=1))
os.makedirs(r'C:\Users\abhishek\submissions\stellar', exist_ok=True)
path = rf'C:\Users\abhishek\submissions\stellar\submission_{score:.4f}.csv'
pd.DataFrame({'id': test['id'], 'class': preds}).to_csv(path, index=False)
print(f'Saved: {path}')
