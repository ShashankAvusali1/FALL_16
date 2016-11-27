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
		# f1_scorer = make_scorer(f1_score)
		for i in range(r):
			for j in range(c):
				grid = GridSearchCV(SVC(kernel='rbf',class_weight={0:weight1[i,j],1:weight2[i,j]}), param_grid=param_grid, cv=cv,scoring='f1')
				grid.fit(X, y)
				# return grid.predict(test_data)
				print("The best parameters are %s , weight1 = %d , weight2 = %d with a score of %0.2f",(grid.best_params_,weight1[i,j],weight2[i,j], grid.best_score_))
				if best_score < grid.best_score_ :
					best_score = grid.best_score_
					best_params = grid.best_params_
					best_params['weight1'] = weight1[i,j]
					best_params['weight2'] = weight2[i,j]
		return best_params 

	def optimize_parameters_rvm(self,X,y,test):
		# C_range = np.logspace(-2,10,13)
		gamma_range = np.logspace(-9,5,13,base=2)
		param_grid = dict(coef1=1./gamma_range)
		cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
		# best_score = 0
		# best_params = {};
		# x1 = np.linspace(0,1,10)
		# y1 = np.linspace(1,20,10)
		# [weight1,weight2] = np.meshgrid(x1,y1)
		# [r,c] = np.shape(weight1)

		# for i in range(r):
		# 	for j in range(1,c):
		# f1_weighted, roc_auc_weighted

		grid = GridSearchCV(RVC(kernel='rbf'), param_grid=param_grid, cv=cv,scoring='f1_weighted')
		grid.fit(X, y)
		print("The best parameters are %s , weight1 = %d , weight2 = %d with a score of %0.2f",(grid.best_params_, grid.best_score_))
		return grid.predict(test)
		# if best_score < grid.best_score_ :
		# 	best_score = grid.best_score_
		# 	best_params = grid.best_params_
		# 	best_params['weight1'] = weight1[i,j]
		# 	best_params['weight2'] = weight2[i,j]
		return grid.best_params_

	def optimize_dos(self,train_data,train_labels):
		train_labels = np.array(train_labels)
		minoritycount = np.count_nonzero(train_labels == 2)
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
		train_data = train_data[:,:19]
		C_range = np.logspace(-2,10,13)
		gamma_range = np.logspace(-9,3,13)
		param_grid = dict(gamma=gamma_range, C=C_range)
		logo = LeaveOneGroupOut()
		best_score = 0
		best_params = {};
		x1 = np.linspace(0,1,5)
		y1 = np.linspace(1,10,5)
		[weight1,weight2] = np.meshgrid(x1,y1)
		[r,c] = np.shape(weight1)
		for i in range(r):
			for j in range(c):
				grid = GridSearchCV(SVC(kernel='rbf',class_weight={1:weight1[i,j],2:weight2[i,j]}), param_grid=param_grid, cv=logo,scoring='f1_weighted')
				grid.fit(train_data, train_labels,groups=groups)
				print("The best parameters are %s ,with a score of %0.2f",(grid.best_params_, grid.best_score_))
				if best_score < grid.best_score_ :
							best_score = grid.best_score_
							best_params = grid.best_params_
							best_params['weight1'] = weight1[i,j]
							best_params['weight2'] = weight2[i,j]
		return best_params
