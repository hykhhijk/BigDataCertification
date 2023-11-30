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