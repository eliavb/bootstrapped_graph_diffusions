import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from random import shuffle
import random


def calcualte_accuracy(nodes_cluster, nodes_to_ignore, predictions):
	num_correct = 0.0
	num_total = 0.0
	if (len(predictions) == 0):
		return 0.0
	for node_id in nodes_cluster:
		if (node_id in nodes_to_ignore):
			continue
		if (node_id in predictions):
			if (nodes_cluster[node_id] == predictions[node_id]):
				num_correct += 1
		num_total += 1
	return num_correct / num_total

def make_high_confidence_predictions(known_labels, cluster_distribution, holdout, \
									 confidence_scores, num_classes, n, args, nodes_cluster):
	num_predictions_to_do = len(nodes_cluster.keys()) * (float(args.percent_to_predict) / 100)
	num_predictions_for_cluster = defaultdict(int)
	labels = list(reversed(sorted(confidence_scores)))
	num_new_predicted_nodes = 0
	num_to_predict_per_cluster = {}
	correct = 0.0
	total = 0.0
	for i in xrange(num_classes):
		num_to_predict_per_cluster[i] = num_predictions_to_do * cluster_distribution[i]
	for t in labels:
		(score, node_id, predicted_cluster) = t
		if (node_id in known_labels or node_id in holdout):
			continue
		if (not node_id in nodes_cluster):
			continue
		if (num_new_predicted_nodes >= num_predictions_to_do):
			continue
		if (num_predictions_for_cluster[predicted_cluster] >= num_to_predict_per_cluster[predicted_cluster]):
			continue
		known_labels[node_id] = predicted_cluster
		num_predictions_for_cluster[predicted_cluster] += 1
		num_new_predicted_nodes += 1
		if (nodes_cluster[node_id] == predicted_cluster):
			correct += 1
	if (num_new_predicted_nodes == 0):
		return num_new_predicted_nodes
	print correct / num_new_predicted_nodes
	return num_new_predicted_nodes

def load_graph(f_name):
	G = nx.Graph()
	with open(f_name, "r") as f:
		for line in f:
			split_line = line.split(",")
			src = int(split_line[0])
			dst = int(split_line[1])
			G.add_edge(src, dst)
	return G


def load_labels(f_name):
	nodes_cluster = {}
	with open(f_name, "r") as f:
		for line in f:
			split_line = line.split(",")
			node_id = int(split_line[0])
			cluster = int(split_line[1])
			nodes_cluster[node_id] = cluster
	return nodes_cluster