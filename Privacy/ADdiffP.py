
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import diffprivlib.models as dp
from sklearn.naive_bayes import GaussianNB
epsilons = [0.1, 0.5, 1, 1.5, 2]



import matplotlib.pyplot as plt

dataset = datasets.load_iris()
# haberman survival dataset

hb = pd.read_csv('C:\\Users\\AmanH\\Downloads\\haberman.csv',names=['Age', 'Year Of Operation 19', 'No. of Positive axillary nodes', 'Survival Status'], header=None)

y = hb['Survival Status']
x = hb.drop('Survival Status', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#bounds = ([min(hb['Age']), min(hb['Year Of Operation 19']), min(hb['No. of Positive axillary nodes']), 1], [max(hb['Age']), max(hb['Year Of Operation 19']), max(hb['No. of Positive axillary nodes']), 2])

min_age = min(hb['Age'])
max_age = max(hb['Age'])
min_year = min(hb['Year Of Operation 19'])
max_year = max(hb['Year Of Operation 19'])
min_number = min(hb['No. of Positive axillary nodes'])
max_number = max(hb['No. of Positive axillary nodes'])

bounds = ([min_age, min_year, min_number], [max_age, max_year, max_number])


nonPrivate_score = GaussianNB()
nonPrivate_score.fit(X_train, y_train)
accuracy_nonPrivate_score = nonPrivate_score.score(X_test, y_test)
print(f"Accuracy without privatization : {accuracy_nonPrivate_score}")
print(f"Acc% without privatization = {accuracy_nonPrivate_score * 100}%")
acc = list()

for e in epsilons:
	clf = dp.GaussianNB(epsilon=e, bounds=bounds)
	clf.fit(X_train, y_train)
	acc.append(clf.score(X_test, y_test))


print(f"Max Acc% after privatization = {max(acc) * 100}%")
print(f"Min acc% after privatization = {min(acc) * 100}%")

import matplotlib.pyplot as plt

plt.title('Differential Privacy using Naive Bayes')
plt.xlabel('epsilons')
plt.ylabel('accuracy')
plt.plot(epsilons, acc)
plt.show()

# diff privacy after adding noise

from diffprivlib.mechanisms import laplace
l = laplace.Laplace()
print(l)
l.set_epsilon(0.05)
l.set_sensitivity(0.02)
l.set_epsilon_delta(0.01, 0.001)

print(l.check_inputs(0))
#scale = self._sensitivity / (self._epsilon - np.log(1 - self._delta))
#rand_val = value - scale * np.sign(unif_rv) * np.log(1 - 2 * np.abs(unif_rv))

#print(l.get_variance(l.randomise(9.0))) #  return 2 * (self._sensitivity / (self._epsilon - np.log(1 - self._delta))) ** 2
r = list()
for i in hb['Age']:
	r.append(l.randomise(i))

hb['Age'] = r
print(r)
x = hb.drop('Survival Status', axis=1)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

acc_laplace = []

for e in epsilons:
	clf = dp.GaussianNB(epsilon=e, bounds=bounds)
	clf.fit(X_train, y_train)
	acc_laplace.append(clf.score(X_test, y_test))

print(f"Max Accuracy after adding noise {max(acc_laplace)}%")
print(f"Min Accuracy after adding noise {min(acc_laplace)}%")

plt.title("Diff privacy after adding noise")
plt.xlabel('epsilons')
plt.ylabel('accuracy')
plt.plot(epsilons, acc_laplace)
plt.show()


'''
hb = pd.read_csv('C:\\Users\\AmanH\\Downloads\\haberman.csv', header=None)
print(hb)


print(x)
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.9)

print( X_train)
print("y_train", y_train)



from sklearn.linear_model import LinearRegression as lr
linear = lr()
md = linear.fit(X_train, y_train)
prediction = md.predict(X_test)
plt.scatter(y_test, prediction)
#plt.show()
print(md.score(X_test, y_test))
from diffprivlib.mechanisms import laplace
#from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
l = laplace.Laplace()
print(l)
l.set_epsilon(0.01)
l.set_sensitivity(0.02)
l.set_epsilon_delta(0.01, 0.001)

print(l.check_inputs(0))
#scale = self._sensitivity / (self._epsilon - np.log(1 - self._delta))
#rand_val = value - scale * np.sign(unif_rv) * np.log(1 - 2 * np.abs(unif_rv))
print(l.randomise(9.0))
print(l.get_variance(l.randomise(9.0))) #  return 2 * (self._sensitivity / (self._epsilon - np.log(1 - self._delta))) ** 2

'''