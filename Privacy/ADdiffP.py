
# TODO
# change the range of epsilons
# change some laplace functions
# try taxonomy trees
# healthcare dataset
# confusion matrix
# what do you conclude
# accuracy parameters
# f1 score, precision, recoil


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import diffprivlib.models as dp
from sklearn.naive_bayes import GaussianNB
epsilons = [0.1, 0.5, 1, 1.5, 2]
#epsilons = [1, 2, 3, 4, 5, 5.7]
#epsilons = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.75, 2, 2.25, 3, 3.5, 4.25, 5.5]

import matplotlib.pyplot as plt

#dataset = datasets.load_iris()
# haberman survival dataset

hb = pd.read_csv('C:\\Users\\AmanH\\Downloads\\haberman.csv',names=['Age', 'Year Of Operation 19', 'No. of Positive axillary nodes', 'Survival Status'], header=None)

y = hb['Survival Status']
x = hb.drop('Survival Status', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

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

"""
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
"""

# FIRST THE LAPLACE NOISE IS ADDED TO THE AGE ATTRIBUTE OF THE DATASET
# THEN THE ORIGINAL ATTRIBUTE IS CHANGED AND REPLACED WITH THE RANODMIZED ONE
# THEN ACCORDINGLY FOR DIFFERENT EPSILONS, THE ABOVE STEPS ARE PERFORMED AND THEN THE DATASET CREATED IS TRAINED
# THEN THE ACCURACY IS CALCULATED

# diff privacy after adding noise

from diffprivlib.mechanisms import laplace
# INITIALIZE LAPLACIAN NOISE

l = laplace.Laplace()

#print(l)


#print(l.check_inputs(0))
#scale = self._sensitivity / (self._epsilon - np.log(1 - self._delta))
#rand_val = value - scale * np.sign(unif_rv) * np.log(1 - 2 * np.abs(unif_rv))

#print(l.get_variance(l.randomise(9.0))) #  return 2 * (self._sensitivity / (self._epsilon - np.log(1 - self._delta))) ** 2

#print(r)
#x = hb.drop('Survival Status', axis=1)


#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

acc_laplace = []
pr = []
pro = []
#var_l = []
# FOR EACH EPSILON THE STEPS ARE PERFORMED
r = list()

x = hb.drop('Survival Status', axis=1)
y = hb['Survival Status']
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

for e in epsilons:
	r_age = []
	#var = []
	clf = dp.GaussianNB(epsilon=e, bounds=bounds)
	l.set_epsilon(e)
	l.set_sensitivity(0.2)
	l.set_epsilon_delta(e, 0)
	for i in X_train['Age']:
		rn = l.randomise(i)
		r_age.append(rn)
	#	var.append(l.get_variance(rn))
	r.append(r_age)
	#var_l.append(var)
	#hb['Age'] = r_age
	X_train['Age'] = r_age
	#X_test['Age'] = r_age
	clf.fit(X_train, y_train)
	pr.append(clf.predict(X_test))
	pro.append(clf.predict_proba(X_test))
	acc_laplace.append(clf.score(X_test, y_test))
	#print(acc_laplace)


# CREATE A LIST OUT OF THE TEST COLUMN (SURVIVAL STATUS OF HABERMAN )

tp_l = []
tn_l = []
fp_l = []
fn_l = []




y_t = []
for i in y_test:
	y_t.append(i)

acc_man_list = []
prec = []
recall = []
f1_score = []
fpr_l = []

for w in range(len(epsilons)):
	prediction = pr[w]
	tp, tn, fp, fn = [], [], [], []
	for z in range(len(prediction)):
		if prediction[z] == 1 and y_t[z] == 1:
			tp.append(prediction[z])
		if prediction[z] == 2 and y_t[z] == 2:
			tn.append(prediction[z])
		if prediction[z] == 1 and y_t[z] != 1:
			fp.append(prediction[z])
		if prediction[z] == 2 and y_t[z] != 2:
			fn.append(prediction[z])
	tp_l.append(tp)
	tn_l.append(tn)
	fp_l.append(fp)
	fn_l.append(fn)

accuracy_parameters = []

for t in range(len(epsilons)):
	#acc, prec1, rec, f1 = [], [], [], []
	#for t in range(len(epsilons)):


	acc_man = (len(tp_l[t]) + len(tn_l[t])) / len(tp_l[t]) + len(tn_l[t]) + len(fp_l[t]) + len(fn_l[t])
	precision = len(tp_l[t])/(len(tp_l[t]) + len(fp_l[t]))
	rec = len(tp_l[t])/(len(tp_l[t]) + len(fn_l[t]))
	f1score = 2*(rec * precision) / (rec + precision)
	fpr = len(fp_l[t]) / (len(fp_l[t]) + len(tn_l[t]))
	#acc.append(acc_man)
	#prec1.append(precision)
	#rec.append(recall)
	#f1.append(f1score)
	fpr_l.append(fpr)
	acc_man_list.append(acc_man)
	prec.append(precision)
	recall.append(rec)
	f1_score.append(f1score)



#print(accuracy_parameters)



#print(k)
print(y_t)

#print( y_test)

# CREATE A PREDICTOR LIST CONSISTING OF THE PREDICTIONS DONE AFTER LAPLACE NOISE

for m in range(len(acc_laplace)):
	if acc_laplace[m] == max(acc_laplace):
		e_m = epsilons[m]
		pr_max = pr[m]
		r_age_max = r[m]
		acc_max = acc_man_list[m]
		prec_max = prec[m]
		recall_max = recall[m]
		f1_score_max = f1_score[m]
		prob_max = pro[m]
	#	var_max = var_l[m]
 		#break
	if acc_laplace[m] == min(acc_laplace):
		e_mi = epsilons[m]
		pr_min = pr[m]
		r_age_min = r[m]
		acc_m = acc_man_list[m]
		prec_m = prec[m]
		recall_m = recall[m]
		f1_score_m = f1_score[m]
		prob_min = pro[m]
	#	var_min = var_l[m]


pr_max_list = list(pr_max)
pr_min_list = list(pr_min)
print("Max acc prediction: ")
print(pr_max_list)
print("Min acc predictions: ")
print(pr_min_list)

print("ACCURACY PARAMS FOR MAX ACC")
print(f"acc {acc_max}")
print(f"prec {prec_max}")
print(f"recall {recall_max}")
print(f"f1score {f1_score_max}")


print("ACCURACY PARAMS FOR MIN ACC")
print(f"acc {acc_m}")
print(f"prec {prec_m}")
print(f"recall {recall_m}")
print(f"f1score {f1_score_m}")



per = 0
for ln in range(len(pr_max_list)):
	if pr_max_list[ln] == y_t[ln]:
		per += 1
a_max = per / len(pr_max_list) * 100
print(f"{a_max}% accuracy manually")

per = 0
for ln_ in range(len(pr_min_list)):
	if pr_min_list[ln_] == y_t[ln_]:
		per += 1
a_min = per / len(pr_min_list) * 100
print(f"{a_min}% manual accuracy")

hb_cp = pd.read_csv('C:\\Users\\AmanH\\Downloads\\haberman.csv',names=['Age', 'Year Of Operation 19', 'No. of Positive axillary nodes', 'Survival Status'], header=None)

print(list(hb_cp['Age']))

print(f"Age privatization variation for max acc" )
print(r_age_max)
print(f"Age privatization variation for min acc")
print(r_age_min)

#print(f"Variance for max acc {var_max}")
#print(f"Variance for min acc {var_min}")



print(f"Max Accuracy for epsilon {e_m} after adding noise {max(acc_laplace) * 100}%")
print(f"Min Accuracy for epsilon {e_mi} after adding noise {min(acc_laplace) * 100}%")

plt.title("Diff privacy after adding noise")

plt.xlabel('epsilons')
plt.ylabel('accuracy')
plt.plot(epsilons, acc_laplace)
#mean_e = sum(epsilons) / len(epsilons)
#mean_a = sum(acc_laplace) / len(acc_laplace)
#plt.scatter(mean_e, mean_a, color="red")
plt.show()

plt.title("Accuracy Graph")
plt.xlabel('epsilons')
plt.ylabel('accuracy')
plt.plot(epsilons, acc_man_list)
plt.show()

plt.title("Precision Graph")
plt.xlabel('epsilons')
plt.ylabel('precision')
plt.plot(epsilons, prec)
plt.show()

plt.title("Recall Graph")
plt.xlabel('epsilons')
plt.ylabel('recall')
plt.plot(epsilons, recall)
plt.show()


plt.title("f1 Score Graph")
plt.xlabel('epsilons')
plt.ylabel('f1 Score')
plt.plot(epsilons, f1_score)
plt.show()



print(acc_man_list)

from sklearn.metrics import roc_curve



def plot_roc_cur(fper, tper, name):
    plt.plot(fper, tper, color='black', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend()
    plt.show()

print(len(tp_l[0]))
print(len(fp_l[0]))
print(len(tn_l[0]))
print(len(fn_l[0]))

print(recall)

prob_M = prob_max
prob_M = prob_M[:, 1]
fpr, tpr, _ = roc_curve(y_test, prob_M, pos_label=1)
plot_roc_cur(fpr, tpr, "ROC For max acc")

prob_mi = prob_min
prob_mi = prob_mi[:, -1]
fpr, tpr, _ = roc_curve(y_test, prob_mi, pos_label=1)
plot_roc_cur(fpr, tpr, "RoC for min")
#print(mean_e, mean_a)

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