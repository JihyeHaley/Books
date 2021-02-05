from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# iris = load_iris()
# dt_clf = DecisionTreeClassifier()
# train_data = iris.data
# print(train_data)
# train_lable  = iris.target
# print(train_lable)
# dt_clf.fit(train_data, train_lable)

# pred = dt_clf.predict(train_data)
# print(f'예측 정확도: {accuracy_score(train_lable, pred)}')

dt_clf = DecisionTreeClassifier()
iris_data = load_iris()

X_train, X_lable, y_train, y_lable = train_test_split(iris_data.data, iris_data.target, train_size=0.8, test_size=0.2, random_state=121)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_lable)

print(f'예측정확도: {accuracy_score(y_lable, pred)}')