import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

class Optimizer(object):
	"""docstring for Optimizer"""
	def __init__(self):
		super(Optimizer, self).__init__()
		
		
	def optimize_parameters(self,X,y):
		C_range = np.logspace(-2,10,13)
		gamma_range = np.logspace(-9,3,13)
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
		best_score = 0
		best_params = {};
		for i in range(1,11):
			grid = GridSearchCV(SVC(kernel='rbf',class_weight={2:i}), param_grid=param_grid, cv=cv)
			grid.fit(X, y)
			print("The best parameters are %s with a score of %0.2f",(grid.best_params_, grid.best_score_))
			if best_score < grid.best_score_ :
				best_score = grid.best_score_
				best_params = grid.best_params_
				best_params['Weight'] = i
		return best_params