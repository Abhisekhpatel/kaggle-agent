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
    df['TicketPrefix'] = df['Ticket'].str.extract(r'([A-Za-z]+)', expand=False).fillna('N')
    df['CabinKnown'] = df['Cabin'].notna().astype(int)
    le = LabelEncoder()
    for col in ['Sex','Embarked','Title','TicketPrefix']:
        df[col] = le.fit_transform(df[col].astype(str))
    keep = ['Pclass','Sex','Age','Fare','FarePerPerson','FamilySize','IsAlone','Title','Embarked','TicketPrefix','CabinKnown']
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

scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'CV: {scores.mean():.4f} +/- {scores.std():.4f}')

model.fit(X, y)

# Pseudo-labelling
proba = model.predict_proba(X_test)
confident = (proba.max(axis=1) > 0.90)
X_pseudo = pd.concat([X, X_test[confident]], ignore_index=True)
y_pseudo = pd.concat([y, pd.Series(proba[confident].argmax(axis=1))], ignore_index=True)
model.fit(X_pseudo, y_pseudo)
print(f'Pseudo-labels added: {confident.sum()}')

preds = model.predict(X_test)
os.makedirs(r'C:\Users\abhishek\submissions', exist_ok=True)
path = rf'C:\Users\abhishek\submissions\submission_v5_pseudo_{scores.mean():.4f}.csv'
pd.DataFrame({'PassengerId': ids, 'Survived': preds.astype(int)}).to_csv(path, index=False)
print(f'Saved: {path}')
