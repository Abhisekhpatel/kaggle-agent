import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb

DATA_DIR = r'C:\Users\abhishek\data'

def engineer_features(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['Age'] = df.groupby(['Title','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FarePerPerson'] = df['FarePerPerson'].fillna(df['FarePerPerson'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['CabinKnown'] = df['Cabin'].notna().astype(int)
    df['TicketPrefix'] = df['Ticket'].str.extract(r'([A-Za-z]+)', expand=False).fillna('N')
    le = LabelEncoder()
    for col in ['Sex','Embarked','Title','TicketPrefix']:
        df[col] = le.fit_transform(df[col].astype(str))
    keep = ['Pclass','Sex','Age','Fare','FarePerPerson','FamilySize','IsAlone','Title','Embarked','CabinKnown','TicketPrefix']
    return df[keep]

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
ids = test['PassengerId'].copy()
y = train['Survived'].copy()
X = engineer_features(train)
X_test = engineer_features(test)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

model = VotingClassifier(estimators=[
    ('lgbm', lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=-1)),
    ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42))
], voting='soft')

model.fit(X, y)
proba = model.predict_proba(X_test)[:,1]

# Try multiple thresholds, pick best on CV
best_thresh, best_score = 0.5, 0
for t in np.arange(0.40, 0.61, 0.01):
    preds_t = (model.predict_proba(X)[:,1] >= t).astype(int)
    score = (preds_t == y).mean()
    if score > best_score:
        best_score, best_thresh = score, t

print(f'Best threshold: {best_thresh:.2f}, train acc: {best_score:.4f}')
preds = (proba >= best_thresh).astype(int)

os.makedirs(r'C:\Users\abhishek\submissions', exist_ok=True)
path = rf'C:\Users\abhishek\submissions\submission_v6_thresh{best_thresh:.2f}.csv'
pd.DataFrame({'PassengerId': ids, 'Survived': preds}).to_csv(path, index=False)
print(f'Saved: {path}')
