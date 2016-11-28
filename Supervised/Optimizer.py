import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sampling import Sampling
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import KFold
from rvm import RVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

class Optimizer(object):
	"""docstring for Optimizer"""
	def __init__(self):
		super(Optimizer, self).__init__()
		self.sampler = Sampling()
		
	def optimize_parameters(self,X,y,test_data):
		C_range = np.logspace(-4,5,13,base=2)
		gamma_range = np.logspace(-9,5,13,base=2)
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
		# cv = StratifiedKFold(n_splits = 10);
		best_score = 0
		best_params = {};
		x1 = np.linspace(0.1,1,5)
		y1 = np.linspace(0.1,2,10)
		[weight1,weight2] = np.meshgrid(x1,y1)
		[r,c] = np.shape(weight1)
		grid = GridSearchCV(SVC(kernel='rbf',class_weight='auto'), param_grid=param_grid, cv=cv,scoring='precision')
		grid.fit(X, y)
		return grid.predict(test_data)


	def optimize_parameters_rvm(self,X,y,test):
		gamma_range = np.logspace(-5,5,13,base=2)
		param_grid = dict(coef1=1./gamma_range)
		cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

		grid = GridSearchCV(RVC(kernel='rbf'), param_grid=param_grid, cv=cv,scoring='precision')
		grid.fit(X, y)
		print("The best parameters are %s , weight1 = %d , weight2 = %d with a score of %0.2f",(grid.best_params_, grid.best_score_))
		return grid.best_params_

	def optimize_dos(self,train_data,train_labels,test_data):
		train_labels = np.array(train_labels)
		minoritycount = np.count_nonzero(train_labels == 1)
		train_sample_count = np.shape(train_data)[0]
		g = np.zeros(train_sample_count-minoritycount)
		kf = KFold(n_splits = minoritycount)
		i = 0
		for train, test in kf.split(g):
			for j in test:
				g[j] = i + 1
			i = i + 1

		g = np.reshape(g,(train_sample_count-minoritycount,1))
		x = [i+1 for i in range(0,minoritycount)]
		x = np.reshape(x,(minoritycount,1))
		g = np.vstack((g,x))
		train_data = np.hstack((train_data,g))

		[train_data,train_labels] = self.sampler.random_over_sampling(train_data,train_labels)
		groups = train_data[:,20]
		train_data = train_data[:,:20]
		C_range = np.logspace(-4,5,13,base=2)
		gamma_range = np.logspace(-9,5,13,base=2)
		param_grid = dict(gamma=gamma_range, C=C_range)
		logo = LeaveOneGroupOut()
		grid = GridSearchCV(SVC(kernel='rbf',class_weight='auto'), param_grid=param_grid, cv=logo,scoring='precision')
		grid.fit(train_data, train_labels,groups=groups)
		return grid.predict(test_data)