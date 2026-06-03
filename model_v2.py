import os, numpy as np, pandas as pd, optuna, warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    le = LabelEncoder()
    for col in ['Sex','Embarked','Title']:
        df[col] = le.fit_transform(df[col].astype(str))
    keep = ['Pclass','Sex','Age','Fare','FarePerPerson','FamilySize','IsAlone','Title','Embarked']
    return df[keep]

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
ids = test['PassengerId'].copy()
y = train['Survived'].copy()
X = engineer_features(train)
X_test = engineer_features(test)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

def tune_lgbm(X, y):
    def obj(trial):
        m = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int('n',50,300),
            max_depth=trial.suggest_int('d',2,6),
            learning_rate=trial.suggest_float('lr',0.01,0.15,log=True),
            num_leaves=trial.suggest_int('nl',10,50),
            min_child_samples=trial.suggest_int('mcs',10,50),
            subsample=trial.suggest_float('ss',0.6,1.0),
            colsample_bytree=trial.suggest_float('cs',0.6,1.0),
            reg_alpha=trial.suggest_float('ra',0,2),
            reg_lambda=trial.suggest_float('rl',0,2),
            random_state=42, verbosity=-1)
        return cross_val_score(m, X, y, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    s = optuna.create_study(direction='maximize')
    s.optimize(obj, n_trials=40)
    print(f'LGBM best CV: {s.best_value:.4f}')
    return s.best_params

lp = tune_lgbm(X, y)
lgbm_model = lgb.LGBMClassifier(
    n_estimators=lp['n'], max_depth=lp['d'], learning_rate=lp['lr'],
    num_leaves=lp['nl'], min_child_samples=lp['mcs'],
    subsample=lp['ss'], colsample_bytree=lp['cs'],
    reg_alpha=lp['ra'], reg_lambda=lp['rl'],
    random_state=42, verbosity=-1)

ensemble = VotingClassifier(estimators=[
    ('lgbm', lgbm_model),
    ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42))
], voting='soft')

scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'Ensemble CV: {scores.mean():.4f} +/- {scores.std():.4f}')

ensemble.fit(X, y)
preds = ensemble.predict(X_test)

os.makedirs(r'C:\Users\abhishek\submissions', exist_ok=True)
sub_path = rf'C:\Users\abhishek\submissions\submission_v2_{scores.mean():.4f}.csv'
pd.DataFrame({'PassengerId': ids, 'Survived': preds.astype(int)}).to_csv(sub_path, index=False)
print(f'Saved to {sub_path}')
