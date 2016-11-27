import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.over_sampling import SMOTE


class Sampling(object):
	"""docstring for Sampling"""
	def __init__(self):
		super(Sampling, self).__init__()
	
	def random_under_sampling(self,data,labels):
		sampler = RandomUnderSampler(ratio = 'auto')
		new_data, new_labels = sampler.fit_sample(data,labels)
		return new_data,new_labels

	def random_over_sampling(self,data,labels):
		sampler = RandomOverSampler(ratio = 'auto')
		new_data, new_labels = sampler.fit_sample(data,labels)
		return new_data,new_labels

	def directed_under_sampling(self, data, labels):
		sampler = CondensedNearestNeighbour()
		new_data, new_labels = sampler.fit_sample(data,labels)
		return new_data,new_labels

	def directed_over_sampling(self, data, labels):
		sampler = SMOTE(ratio = 'auto', k_neighbors = 3)
		new_data, new_labels = sampler.fit_sample(data,labels)
		return new_data,new_labels
