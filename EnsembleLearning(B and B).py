from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

test_size = float(input("Enter test size (e.g. 0.2): "))
n_estimators = int(input("Enter number of estimators: "))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=n_estimators)
bag.fit(X_train, y_train)

boost = AdaBoostClassifier(n_estimators=n_estimators)
boost.fit(X_train, y_train)

bag_pred = bag.predict(X_test)
boost_pred = boost.predict(X_test)

print("Bagging Accuracy:", accuracy_score(y_test, bag_pred))
print("Boosting Accuracy:", accuracy_score(y_test, boost_pred))
