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
from rvm import RVC


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
	train_labels = np.vstack((np.ones((majority_train,1)),np.ones((minority_count,1))*2))
	train_labels = np.reshape(train_labels,majority_train+minority_train)

	test_data = np.vstack((class1[majority_train:],class2[100:]))
	test_labels = np.vstack((np.ones((majority_test,1)),np.ones((minority_test,1))*2))
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

# sampler = Sampling()

optimizer = Optimizer()
best_params = optimizer.optimize_parameters_rvm(train_data, train_labels)


rvc = RVC(kernel='rbf',coef1 = best_params['coef1'])
rvc.fit(train_data, train_labels)
prob = rvc.predict_proba(train_data)
y_pred = rvc.predict(test_data)
display_results(y_pred, test_labels)

