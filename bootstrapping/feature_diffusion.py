import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import diags
import time
import sys

import base_bootstrapped_algo
import base_predictor
from sklearn import multiclass
from sklearn import linear_model


class BaseFeatureDiffusion(base_predictor.BasePredictor):
	def __init__(self, G, num_classes, seed_nodes_to_cluster, nodes_to_features, args):
		self.G = G
		self.num_classes = num_classes
		self.num_nodes = len(self.G)
		self.seed_nodes_to_cluster = seed_nodes_to_cluster.copy()
		self.nodes_to_features = nodes_to_features.copy()
		self.feature_vector_size = 0
		for node_id in self.nodes_to_features:
			self.feature_vector_size = self.nodes_to_features[node_id].shape[1]
		self.args = args
		self.y = np.zeros((self.num_nodes, self.feature_vector_size), dtype=np.float32)
		self.set_features()

	def add_labeled_nodes(self, seed_nodes_to_cluster):
		self.seed_nodes_to_cluster = seed_nodes_to_cluster.copy()

	def set_features(self):
		for node_id in sorted(self.nodes_to_features.keys()):
			v = self.nodes_to_features[node_id].todense()
			self.y[node_id, :] = np.squeeze( np.asarray(v) ) 

	def fit(self):
		pass

	def get_predict_probs(self):
		one_vs_rest = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
		train_feature_vectors = []
		train_classes = []
		for node_id in sorted(self.seed_nodes_to_cluster.keys()):
			train_feature_vectors.append(self.y[node_id, :])
			train_classes.append(self.seed_nodes_to_cluster[node_id])
		one_vs_rest.fit(train_feature_vectors, train_classes)
		test_features = []
		node_ids = []
		for node_id in sorted(self.nodes_to_features.keys()):
			node_ids.append(node_id)
			test_features.append(self.y[node_id, :])
		predict_proba = one_vs_rest.predict_proba(test_features)
		confidence_scores = []
		for i, node_id in enumerate(node_ids):
			class_to_predict = np.argmax(predict_proba[i])
			score = np.max(predict_proba[i])
			confidence_scores.append((score, node_id, class_to_predict))
		return confidence_scores

	def get_confidence_scores(self):
		return self.get_predict_probs()

	def predict(self):
		predictions = {}
		most_confident_classes = self.get_confidence_scores()
		for tup in most_confident_classes:
			score, node_id, class_to_predict = tup
			predictions[node_id] = class_to_predict
		return predictions

class NonNormalizedFeatureDiffusion(BaseFeatureDiffusion):
	def __init__(self, G, M, num_classes, seed_nodes_to_cluster, nodes_to_features, args):
		super(NonNormalizedFeatureDiffusion, self).__init__(G, num_classes, seed_nodes_to_cluster, nodes_to_features, args)
		self.M = M
		self.did_fit = False

	def fit(self):
		if (not self.did_fit):
			for k in xrange(self.args.iterations):
				self.y = self.M.dot(self.y)
			self.did_fit = True

class NormalizedFeatureDiffusion(BaseFeatureDiffusion):
	def __init__(self, G, M, num_classes, seed_nodes_to_cluster, nodes_to_features, args):
		super(NormalizedFeatureDiffusion, self).__init__(G, num_classes, seed_nodes_to_cluster, nodes_to_features, args)
		self.M = M
		self.return_prob = args.return_prob
		self.y_return = self.y.copy() * self.return_prob
		self.did_fit = False

	def fit(self):
		if (not self.did_fit):
			for k in xrange(self.args.iterations):
				self.y = self.y_return + (1-self.return_prob) * self.M.dot(self.y)
			self.did_fit = True

	def add_labeled_nodes(self, seed_nodes_to_cluster):
		super(NormalizedFeatureDiffusion, self).add_labeled_nodes(seed_nodes_to_cluster)
		self.y_return = self.y.copy() * self.return_prob

class FeatureDiffusionBootstrapped(base_bootstrapped_algo.BaseBootstrapped):
	def __init__(self, G, num_classes, seed_nodes_to_cluster, args, special_params):
		super(FeatureDiffusionBootstrapped, self).__init__(G, num_classes, seed_nodes_to_cluster, args, special_params)
		if (args.model == "feature_diffusion_norm_lp"):
			self.bootstrapped_predictor = NormalizedFeatureDiffusion(G, special_params["M"], num_classes, seed_nodes_to_cluster, special_params["features"], args)
			self.base_predictor = NormalizedFeatureDiffusion(G, special_params["M"], num_classes, seed_nodes_to_cluster, special_params["features"], args)
		if (args.model == "feature_diffusion_lp"):
			self.bootstrapped_predictor = NonNormalizedFeatureDiffusion(G, special_params["M"], num_classes, seed_nodes_to_cluster, special_params["features"], args)
			self.base_predictor = NonNormalizedFeatureDiffusion(G, special_params["M"], num_classes, seed_nodes_to_cluster, special_params["features"], args)