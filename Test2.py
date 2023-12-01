import numpy as np
import pandas as pd

#1유형
#Attrition이 종속변수인데 Yes:1, No:0으로 인코딩하고 범주변 레코드 수 계산
#데이터셋 컬럼개수 계산, 범주형 중 한가지 값만 가지고 있는 컴럼있는데 그거 제거
#숫자형 컬럼만 추출해 새로운 데이터셋 생성, 변수가 상관관계 구하고 0.9이상인 두개의 컬럼을 찾아내 그 중 하나 제거
df = pd.read_csv("./data/attrition.csv")
y_index = df["Attrition"]=="Yes"
n_index = df["Attrition"]=="No"
df.loc[y_index,"Attrition"] = 1
df.loc[n_index, "Attrition"] = 0
print(df["Attrition"].value_counts())

print(len(df.columns))
print(df.describe(include="object").loc["unique"])
df = df.drop("Over18", axis=1)
num_columns = []
for i in df.columns:
    if df[i].dtype=="int64":
        num_columns.append(i)

#####       DataFrame.select_dtypes() chk

over9 = []
for row in num_columns:
    for col in num_columns:
        if row==col:
            continue
        else:
            if df[num_columns].corr()[row][col] > 0.9:
                over9.append([row,col])
print(over9)
df = df.drop("JobLevel", axis=1)
print(len(df.columns))


#2유형
#파킨슨 데이터 로지스틱 회귀
#가장 영향력 높은 변수 3개 순서대로 선정
#threshold를 0.5와 0.8로 했을때 f1score비교
#name 변수 제거, min-max scaler 사용, LR을 위해 상수항 추가?, Status 카테고리타입 변환, train:test 9:1, bfgs algorithm 사용
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df2 = pd.read_csv("./data/parkinsons.txt")
df2 = df2.drop("name", axis=1)

X = df2[[i for i in df2.columns if i != "status"]]
y = df2["status"]           #df2["status"].astype("category") 느낌으로 category형으로 변환 가능...
X_train, X_test, y_train, y_test = train_test_split(X.copy(), y.copy(), test_size=0.1)
for column in X_train.columns:
    scaler = MinMaxScaler()
    X_train[column] = scaler.fit_transform(X_train[[column]])
    X_test[column] = scaler.transform(X_test[[column]])
model = LogisticRegression(solver="lbfgs")   ##########
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
over5 = proba[:,1] > 0.5
over8 = proba[:,1] > 0.8
# print(model.p_value)
print(f1_score(y_test, over5))
print(f1_score(y_test, over8))