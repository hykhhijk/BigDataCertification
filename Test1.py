##############################################################################
#라이브러리 모듈확인법
from sklearn.neighbors import __all__ as cluster_modules

print("Modules in sklearn.cluster:")
for module in cluster_modules:
    print(f"- {module}")
##############################################################################


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
without_z = np.mean(df[df["Ozone"]!=0]["Ozone"])    #결측치가 아닌 0값이 존재한다면 오답이 될 수 있을듯 -> 미리 Ozone의 값 범위를 확인 후 평균구하기
print(abs(with_z - without_z))

#2
print(df["Wind"])
scaler = MinMaxScaler()
minmax = scaler.fit_transform(df[["Wind"]])
zscore = StandardScaler().fit_transform(df[["Wind"]])
print(minmax.mean())
print(zscore.mean())
print(abs(minmax.mean() - zscore.mean()))       #넘파이와 판다스의 소수점 자리수 차이로 정답과 오차 -> 소수점 몇째자리까지 계산해야 하는거지? ->출력형태가 정해져있던거 같은데 체크할것

#3
temps = []
for i in range(5, 10):
    temps.append(np.mean(df[df["Month"]==i]["Temp"].mean()))
print(temps)



#2유형
#은행데이터의 대출여부를 분류하는 최족의 이웃 크기값(k)을 구하고 분류정확도 산출
#7:3으로 split하고 stratifed, normalizer를 이용하여 scaling

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df2 = pd.read_csv("./data/Bank_Personal_Loan_Modelling.csv")
print(df2)      #label: Personal Loan
X =df2[[i for i in df2.columns if i!="Personal Loan"]]
y = df2[["Personal Loan"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)
labels = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]
#문제가 normalizer를 사용해 scaling 하라는거라서 StandardScaler를 시용했는데
#sklearn.preprocessing에 processing.Normalizer()를 사용하면 행별 l1, l2 정규화가 가능하다
#->normalizer 혹은 데이터 정규화할때 이점을 유의해야할듯
#Normalizer의 param으로 norm={"l1","l2"}로 선택가능
#fit_transform할때는 데이터 프레임을 한번에 넣어도 된다
for i in labels:    
    scaler = StandardScaler()        
    scaler.fit(X_train[[i]])
    X_train.loc[:][i] = scaler.transform(X_train[[i]])
    X_test.loc[:][i] = scaler.transform(X_test[[i]])

cnt = [0]
for k in range(1, 15):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cnt.append(np.sum(pred==np.array(y_test).reshape(-1)) / len(y_test))
print(cnt)
k = np.argmax(cnt)
print(cnt[k] * 100) #많이 못생겼는데 소수점 아래 자르고 출력하는거 꼭 외워가자