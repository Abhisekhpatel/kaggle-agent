import os, json, numpy as np, pandas as pd, optuna, warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['Age'] = df.groupby(['Title','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=['Child','Teen','Adult','Middle','Senior'])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FarePerPerson'] = df['FarePerPerson'].fillna(df['FarePerPerson'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    le = LabelEncoder()
    for col in ['Sex','Embarked','Title','Deck','AgeBin']:
        df[col] = le.fit_transform(df[col].astype(str))
    df = df.drop(columns=['Name','Ticket','Cabin','PassengerId'], errors='ignore')
    return df

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
ids = test['PassengerId'].copy()
y = train['Survived'].copy()
X = engineer_features(train.drop(columns=['Survived']))
X_test = engineer_features(test).reindex(columns=X.columns, fill_value=0)
print(f'Features: {list(X.columns)}')

def tune_lgbm(X, y):
    def obj(trial):
        m = lgb.LGBMClassifier(n_estimators=trial.suggest_int('n',100,500), max_depth=trial.suggest_int('d',3,8), learning_rate=trial.suggest_float('lr',0.01,0.2,log=True), num_leaves=trial.suggest_int('nl',20,80), subsample=trial.suggest_float('ss',0.6,1.0), colsample_bytree=trial.suggest_float('cs',0.6,1.0), random_state=42, verbosity=-1)
        return cross_val_score(m, X, y, cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring='accuracy', n_jobs=-1).mean()
    s = optuna.create_study(direction='maximize')
    s.optimize(obj, n_trials=30)
    print(f'LGBM best: {s.best_value:.4f}')
    return s.best_params

def tune_xgb(X, y):
    def obj(trial):
        m = xgb.XGBClassifier(n_estimators=trial.suggest_int('n',100,500), max_depth=trial.suggest_int('d',3,7), learning_rate=trial.suggest_float('lr',0.01,0.2,log=True), subsample=trial.suggest_float('ss',0.6,1.0), colsample_bytree=trial.suggest_float('cs',0.6,1.0), random_state=42, verbosity=0, eval_metric='logloss')
        return cross_val_score(m, X, y, cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring='accuracy', n_jobs=-1).mean()
    s = optuna.create_study(direction='maximize')
    s.optimize(obj, n_trials=20)
    print(f'XGB best: {s.best_value:.4f}')
    return s.best_params

lp = tune_lgbm(X, y)
xp = tune_xgb(X, y)

ensemble = VotingClassifier(estimators=[
    ('lgbm', lgb.LGBMClassifier(**{k.replace('n','n_estimators').replace('d','max_depth').replace('lr','learning_rate').replace('nl','num_leaves').replace('ss','subsample').replace('cs','colsample_bytree'):v for k,v in lp.items()}, random_state=42, verbosity=-1)),
    ('xgb', xgb.XGBClassifier(**{k.replace('n','n_estimators').replace('d','max_depth').replace('lr','learning_rate').replace('ss','subsample').replace('cs','colsample_bytree'):v for k,v in xp.items()}, random_state=42, verbosity=0)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42))
], voting='soft')

cv = cross_val_score(ensemble, X, y, cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring='accuracy', n_jobs=-1)
print(f'Ensemble CV: {cv.mean():.4f} +/- {cv.std():.4f}')
ensemble.fit(X, y)
preds = ensemble.predict(X_test)

os.makedirs(r'C:\Users\abhishek\submissions', exist_ok=True)
sub_path = rf'C:\Users\abhishek\submissions\submission_{cv.mean():.4f}.csv'
pd.DataFrame({'PassengerId': ids, 'Survived': preds.astype(int)}).to_csv(sub_path, index=False)
print(f'Done! Submission saved to {sub_path}')
