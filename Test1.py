import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1유형
#1.ari quality데이터 결측치가 가장 많은 변수 탐색, 결측치 0으로 대치, 결측치 제외 평균과 0대치 평균의 차이 구하기
#2. Wind 변수 minmax변환후 평균값, Z정규화 후 평균값 이 둘의 차이
#3. 월별(5월~9월) 평균기온

#1
df = pd.read_csv("./data/airquality.csv")
df["Ozone"] = df["Ozone"].fillna(0)
with_z = np.mean(df["Ozone"])
without_z = np.mean(df[df["Ozone"]!=0]["Ozone"])
print(abs(with_z - without_z))

#2
print(df["Wind"])
scaler = MinMaxScaler()
minmax = scaler.fit_transform(df[["Wind"]])
zscore = StandardScaler().fit_transform(df[["Wind"]])
# print(minmax.mean())
# print(zscore.mean())
print(abs(minmax.mean() - zscore.mean()))

#3
temps = []
for i in range(5, 10):
    temps.append(np.mean(df[df["Month"]==i]["Temp"].mean()))
print(temps)



#2유형
from sklearn.neighbors import __all__ as cluster_modules

print("Modules in sklearn.cluster:")
for module in cluster_modules:
    print(f"- {module}")


import sklearn.cluster
print(sklearn.cluster.packages)
df2 = pd.read_csv("./data/Bank_Personal_Loan_Modelling.csv")
