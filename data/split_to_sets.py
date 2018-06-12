import random
import os
from collections import defaultdict as dd
import itertools
import numpy as np

def read_labels(labels_path):
	node_id_to_class = {}
	with open(labels_path) as f_labels:
		for line in f_labels:
			split_line = line.split(",")
			node_id = int(split_line[0])
			label = int(split_line[1])
			node_id_to_class[node_id] = label
	return node_id_to_class

def split_to_sets(node_id_to_class, seed_set_size, holdout_size):
	node_ids = node_id_to_class.keys()
	np.random.shuffle(node_ids)
	num_classes = len(list(set(node_id_to_class.values())))
	seed_set = {}
	holdout = {}
	num_nodes_per_cluster = dd(int)
	for node_id in range(seed_set_size):
		node_class = node_id_to_class[node_id]
		if (num_nodes_per_cluster[node_class] >= seed_set_size):
			continue
		seed_set[node_id] = node_class
		num_nodes_per_cluster[node_class] += 1
	for node_id in node_ids:
		node_class = node_id_to_class[node_id]
		if (node_id in seed_set):
			continue
		if (len(holdout) >= holdout_size):
			break
		holdout[node_id] = node_class
	assert len(set(seed_set.keys()).intersection(holdout.keys())) == 0
	return seed_set, holdout

def split_to_sets_random(node_id_to_class, seed_set_size, holdout_size):
	node_ids = node_id_to_class.keys()
	np.random.shuffle(node_ids)
	num_classes = len(list(set(node_id_to_class.values())))
	seed_set = {}
	holdout = {}
	num_nodes_per_cluster = dd(int)
	for node_id in node_ids:
		node_class = node_id_to_class[node_id]
		if (num_nodes_per_cluster[node_class] >= seed_set_size):
			continue
		if (num_nodes_per_cluster[node_class] >= seed_set_size / num_classes):
			continue
		seed_set[node_id] = node_class
		num_nodes_per_cluster[node_class] += 1

	for node_id in node_ids:
		node_class = node_id_to_class[node_id]
		if (node_id in seed_set):
			continue
		if (len(holdout) >= holdout_size):
			break
		holdout[node_id] = node_class
	assert len(set(seed_set.keys()).intersection(holdout.keys())) == 0
	return seed_set, holdout

def dump_to_dir(dir_path, seed_set, holdout, node_id_to_class):
	holdout_path = os.path.join(dir_path, "holdout")
	seed_set_path = os.path.join(dir_path, "seed_set")
	all_seeds_path = os.path.join(dir_path, "all_labels")
	with open(holdout_path, "w") as holdout_f:
		for node_id, node_class in holdout.iteritems():
			holdout_f.write(str(node_id) + "," + str(node_class) + "\n")
	with open(seed_set_path, "w") as seed_set_f:
		for node_id, node_class in seed_set.iteritems():
			seed_set_f.write(str(node_id) + "," + str(node_class) + "\n")
	with open(all_seeds_path, "w") as all_seeds_f:
		for node_id, node_class in node_id_to_class.iteritems():
			all_seeds_f.write(str(node_id) + "," + str(node_class) + "\n")

def create_dir_if_needed(dir_path):
	try:
		os.makedirs(dir_path)
	except Exception as e:
		print "Problem", str(e)

def create_random_sets():
	data_sets =  ["citeseer", "pubmed", "cora"]
	holdout_sizes = [500, 500, 500]
	seed_set_sizes = [120, 60, 140]
	num_sets_to_create = 100
	for data_set, holdout_size, seed_set_size in zip(data_sets, holdout_sizes, seed_set_sizes):
		for i in xrange(num_sets_to_create):
			labels_path = "./%s/%s_random_splits/labels/labels" % (data_set, data_set)
			print labels_path
			node_id_to_class = read_labels(labels_path)
			seed_set, holdout = split_to_sets_random(node_id_to_class, seed_set_size, holdout_size)
			path_to_dir = "./%s/%s_random_splits/set%s" % (data_set, data_set, i)
			create_dir_if_needed(path_to_dir)
			dump_to_dir(path_to_dir, seed_set, holdout, node_id_to_class)


if __name__ == '__main__':
	create_random_sets()