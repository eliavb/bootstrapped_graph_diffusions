import networkx as nx
import cPickle
import numpy as np
from scipy.sparse import csr_matrix

def get_graph(graph_file_path):
	G = nx.Graph()
	node_name_to_node_id = {}
	current_open_id = 0
	with open(graph_file_path) as f_graph:
		for line in f_graph:
			split_line = line.split("\t")
			src = split_line[0].strip()
			dst = split_line[1].strip()
			if (not src in node_name_to_node_id):
				node_name_to_node_id[src] = current_open_id
				current_open_id += 1
			if (not dst in node_name_to_node_id):
				node_name_to_node_id[dst] = current_open_id
				current_open_id += 1
			src = node_name_to_node_id[src]
			dst = node_name_to_node_id[dst]
			G.add_edge(src, dst)
	print "G.number_of_nodes()", G.number_of_nodes()
	print "G.number_of_edges()", G.number_of_edges()
	return G, node_name_to_node_id

def get_labels(label_file_path, node_name_to_node_id):
	node_id_to_label = {}
	label_name_to_label_id = {}
	current_open_label_id = 0
	node_id_to_feature_vector = {}
	with open(label_file_path) as f_label:
		for line in f_label:
			split_line = line.split("\t")
			node_id = split_line[0].strip()
			node_id = node_name_to_node_id[node_id]
			label_id = split_line[-1].strip()
			feature = np.array(split_line[1:-1], dtype=np.int32)
			if (not label_id in label_name_to_label_id):
				label_name_to_label_id[label_id] = current_open_label_id
				current_open_label_id += 1
			label_id = label_name_to_label_id[label_id]
			node_id_to_label[node_id] = label_id
			node_id_to_feature_vector[node_id] = csr_matrix(feature)
	return node_id_to_label, node_id_to_feature_vector

def main():
	graph_file_path = "./citeseer.cites"
	label_file_path = "./citeseer.content"
	G, node_name_to_node_id = get_graph(graph_file_path)
	node_id_to_label, node_id_to_feature_vector = get_labels(label_file_path, node_name_to_node_id)
	print len(node_id_to_label)
	print set(node_id_to_label.values())
	nx.write_edgelist(G, "./graph/graph.csv", delimiter=",", data=False)
	with open("./labels/labels", "w") as f_labels:
		for node_id, label in node_id_to_label.iteritems():
			f_labels.write(str(node_id) + "," + str(label) + "\n")
	cPickle.dump(node_id_to_feature_vector, open("./features/features", "w"))

if __name__ == '__main__':
	main()