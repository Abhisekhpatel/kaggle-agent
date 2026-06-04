import os, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

DATA_DIR = r'C:\Users\abhishek\data'

def features(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['Age'] = df.groupby(['Title','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize']==1).astype(int)
    le = LabelEncoder()
    for c in ['Sex','Embarked','Title']:
        df[c] = le.fit_transform(df[c].astype(str))
    return df[['Pclass','Sex','Age','Fare','FamilySize','IsAlone','Title','Embarked']]

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
ids = test['PassengerId']
y = train['Survived']
X = features(train)
X_test = features(test)

m = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
cv = cross_val_score(m, X, y, cv=10, scoring='accuracy').mean()
print(f'LR CV: {cv:.4f}')
m.fit(X, y)
preds = m.predict(X_test)
os.makedirs(r'C:\Users\abhishek\submissions', exist_ok=True)
path = rf'C:\Users\abhishek\submissions\submission_v7_{cv:.4f}.csv'
pd.DataFrame({'PassengerId': ids, 'Survived': preds}).to_csv(path, index=False)
print(f'Saved: {path}')
