import sklearn

# print(sklearn.__version__)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
'''
    데이터 세트를 학습 데이터와 테스트 데이터로 분리하는 train_test_split()함수
    test_size = 0.2로 설정하면, 20%는 테스트 데이터, 80%는 학습데이터
'''
from sklearn.model_selection import train_test_split 

import pandas as pd

iris = load_iris()

# iris.data는 Iris 데이터 세트에서 피처(Feature)만으로 된 데이터를 numpy로 가지고 있다.
iris_data = iris.data #numpy 데이터
print(iris_data)

# iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있다.
iris_label = iris.target # 결정값 = label
print(f'iris target 값: {iris_label}')
print(f'iris target 명: {iris.target_names}')

# 붗꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다.
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df.head(3)

# random_state는 호출할 때마다 같은 학습/테스트용 데이터를 생성하기 위해 주어지는 난수 발생 값..
''' random_state값이 없다면, 매번 다른데이터를 가져와서 문제가 많을 것이다.'''
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, train_size=0.8, }random_state=11)
print(f'X_test {X_test}')
print(f'X_test:type {type(X_test)}')
'''
학습용 피처 데이터 세트를 X_train # 80% 
테스트용 피처 데이터 세트를 X_test # 20%
학습용 레이블 데이터 세트를 y_train  # 80% 
테스트용 레이블 데이터 세트를 y_test # 20%
'''

# DecisionTreeClassifier 객체 생성
''' 생성된 DecisionTreeClassifier 객체의 fit() 메서드에 학습용 피처 데이터 속석과 결정값 데이터 세트를 입력해 호출하면 학습을 수행한다.'''
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)  # 학습시킴
''' 객체는 학습 데이터를 기반으로 학습이 완료!, 이렇게완료된 학습 객체를 이용해서 예측을 수행
  ***** 예측은 반드시 학습데이터가 아닌 다르 데이터를 이용해야한다. (일반적으로 테스트 데이터 세트를 이용)
  DecisionTreeClassifier 객체의 predit() 메서드에 테스트용 피처 데이터 세트를 입력해 호출하면 학습된 모델 기반에서 테스트 데이터 세트에 대한 예측값을 반환한다.
'''

pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(f'예측 정확도: {accuracy_score(y_test, pred)}')

'''
1. 데이터 세트 분리: 데이터를 학습 데이터와 테스트 데이터로 분리합니다.
2. 모델학습: 학습 데이터를 기반으로 ML 알고리즘을 적용해 모델을 학습시킵니다.
3. 예측수행: 학습된 ML모델을 이용해 테스트 데이터의 분류(즉, 붓꽃 종류)를 예측합니다.
4. 평가: 이렇게 예측된 결괏값과 테스트 데이터의 실제 결괏값을 비교해 ML 모델 성능을 평가합니다. 
'''