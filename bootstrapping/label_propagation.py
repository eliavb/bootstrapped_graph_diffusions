import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import time
import sys
import base_bootstrapped_algo
import base_predictor

def get_inverse_diagonal(W):
	D = np.array(W.sum(axis=0))[0, :]
	inverse = []
	for val in D:
		if (val != 0):
			inverse.append(1 / float(val))
		else:
			inverse.append(0.0)
	D = diags([inverse], [0])
	return D

def get_square_root_inverse_diagonal(W):
	D = np.array(W.sum(axis=0))[0, :]
	inverse = []
	for val in D:
		if (val != 0):
			inverse.append(1 / np.sqrt(float(val)))
		else:
			inverse.append(0.0)
	D = diags([inverse], [0])
	return D

def get_graph_laplacian(G):
	W = csr_matrix(nx.to_scipy_sparse_matrix(G), dtype=np.float32)
	D_inverse = csr_matrix(get_inverse_diagonal(W), dtype=np.float32)
	M = D_inverse.dot(W)
	return M

def get_graph_normalized_laplacian(G):
	W = csr_matrix(nx.to_scipy_sparse_matrix(G), dtype=np.float32)
	D_inverse = csr_matrix(get_square_root_inverse_diagonal(W), dtype=np.float32)
	M = D_inverse.dot(W.dot(D_inverse))
	return M

class BaseLabelPropagation(base_predictor.BasePredictor):
	def __init__(self, G, num_classes, seed_nodes_to_cluster, args):
		self.G = G
		self.num_classes = num_classes
		self.num_nodes = len(self.G)
		self.seed_nodes_to_cluster = seed_nodes_to_cluster.copy()
		self.args = args
		self.y = np.zeros((self.num_nodes, self.num_classes), dtype=np.float32)
		self.set_seed_set_entries(weight=1)

	def add_labeled_nodes(self, seed_nodes_to_cluster):
		self.seed_nodes_to_cluster = seed_nodes_to_cluster.copy()
		self.y = np.zeros((self.num_nodes, self.num_classes), dtype=np.float32)
		self.set_seed_set_entries()

	def set_seed_set_entries(self, weight=1):
		minus_weight_row = [-weight] * self.num_classes
		node_ids = []
		clusters = []
		for node_id, cluster in self.seed_nodes_to_cluster.iteritems():
			node_ids.append(node_id)
			clusters.append(cluster)
		self.y[node_ids, :] = minus_weight_row
		self.y[node_ids, clusters] = weight

	def fit(self):
		pass

	def get_confidence_scores(self):
		confidence_scores = []
		for node_id in xrange(self.num_nodes):
			score = np.amax(self.y[node_id, :])
			cluster = np.argmax(self.y[node_id, :])
			confidence_scores.append((score, node_id, cluster))
		return confidence_scores

	def predict(self):
		predictions = {}
		for node_id in xrange(self.num_nodes):
			cluster = np.argmax(self.y[node_id, :])
			predictions[node_id] = cluster
		return predictions

class NonNormalizedLabelPropagation(BaseLabelPropagation):
	def __init__(self, G, M, num_classes, seed_nodes_to_cluster, args):
		super(NonNormalizedLabelPropagation, self).__init__(G, num_classes, seed_nodes_to_cluster, args)
		self.M = M

	def fit(self):
		for k in xrange(self.args.iterations):
			self.set_seed_set_entries()
			self.y = self.M.dot(self.y)

class NormalizedLabelPropagation(BaseLabelPropagation):
	def __init__(self, G, M, num_classes, seed_nodes_to_cluster, args):
		super(NormalizedLabelPropagation, self).__init__(G, num_classes, seed_nodes_to_cluster, args)
		self.M = M
		self.return_prob = args.return_prob
		self.y_return = self.y.copy() * self.return_prob

	def fit(self):
		for k in xrange(self.args.iterations):
			self.y = self.y_return + (1-self.return_prob) * self.M.dot(self.y)

	def add_labeled_nodes(self, seed_nodes_to_cluster):
		super(NormalizedLabelPropagation, self).add_labeled_nodes(seed_nodes_to_cluster)
		self.y_return = self.y.copy() * self.return_prob

class LabelPropagationBootstrapped(base_bootstrapped_algo.BaseBootstrapped):
	def __init__(self, G, num_classes, seed_nodes_to_cluster, args, special_params):
		super(LabelPropagationBootstrapped, self).__init__(G, num_classes, seed_nodes_to_cluster, args, special_params)
		if (args.model == "norm_lp"):
			self.bootstrapped_predictor = NormalizedLabelPropagation(G, special_params["M"], num_classes, seed_nodes_to_cluster, args)
			self.base_predictor = NormalizedLabelPropagation(G, special_params["M"], num_classes, seed_nodes_to_cluster, args)
		if (args.model == "lp"):
			self.bootstrapped_predictor = NonNormalizedLabelPropagation(G, special_params["M"], num_classes, seed_nodes_to_cluster, args)
			self.base_predictor = NonNormalizedLabelPropagation(G, special_params["M"], num_classes, seed_nodes_to_cluster, args)