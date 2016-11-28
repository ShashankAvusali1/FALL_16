import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import ClusterCentroids
from Optimizer import Optimizer
from sampling import Sampling
from sklearn.model_selection import KFold


def display_results(y_pred,y_true):
	result  = y_pred.astype(int) == y_true.astype(int)
	
	print('===== Overall result =====')
	print(np.bincount(result))
	print('====== Minority class result =====')
	result = y_pred[200:] == y_true[200:]
	print(np.bincount(result))

def split_train_test(data, labels, minority_count):

	minority_train = minority_count
	minority_test = 200
	majority_test = 200
	majority_train = 500
	class1 = data[np.where(labels== 1)]
	class2 = data[np.where(labels == 2)]

	train_data = np.vstack((class1[:majority_train],class2[:minority_train]))
	train_labels = np.vstack((np.zeros((majority_train,1)),np.ones((minority_count,1))))
	train_labels = np.reshape(train_labels,majority_train+minority_train)

	test_data = np.vstack((class1[majority_train:],class2[100:]))
	test_labels = np.vstack((np.zeros((majority_test,1)),np.ones((minority_test,1))))
	test_labels = np.reshape(test_labels,minority_test+majority_test)

	print('####')
	print('shape of train_data')
	print(np.shape(train_data))
	print('shape of test data')
	print(np.shape(test_data))

	return train_data, train_labels, test_data, test_labels


np.random.seed(100)

data = np.loadtxt('data/german.data-numeric.txt')
data = np.array(data);
scaler  = StandardScaler()
labels = data[:,24].astype(int)
data = scaler.fit_transform(data[:,:20])

minoritycount = 20

[train_data, train_labels, test_data, test_labels] = split_train_test(data,labels,minority_count=minoritycount)

sampler = Sampling()


# [train_data,train_labels] = sampler.random_under_sampling(train_data,train_labels)

#[train_data,train_labels] = sampler.random_over_sampling(train_data,train_labels)

# [train_data, train_labels] = sampler.directed_under_sampling(train_data, train_labels)


optimizer = Optimizer()
#y_pred = optimizer.optimize_parameters(train_data,train_labels,test_data)
y_pred = optimizer.optimize_dos(train_data, train_labels,test_data)

# svc = SVC(C=bestparams['C'], kernel='rbf', gamma=bestparams['gamma'], shrinking=True, probability=False, class_weight='auto', tol=0.001, verbose=False)
# svc.fit(train_data,train_labels)
# y_pred = svc.predict(test_data)
display_results(y_pred,test_labels)


