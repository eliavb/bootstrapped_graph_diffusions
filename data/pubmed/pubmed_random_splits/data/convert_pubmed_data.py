import networkx as nx
import cPickle

def get_graph(graph_file_path):
	G = nx.Graph()
	node_label_to_node_id = {}
	current_open_node_id = 0
	with open(graph_file_path) as f_graph:
		next(f_graph)
		next(f_graph)
		for line in f_graph:
			split_line = line.split("\t")
			src = int(split_line[1].split("paper:")[1])
			dst = int(split_line[3].split("paper:")[1])
			if (not src in node_label_to_node_id):
				node_label_to_node_id[src] = current_open_node_id
				current_open_node_id += 1
			if (not dst in node_label_to_node_id):
				node_label_to_node_id[dst] = current_open_node_id
				current_open_node_id += 1
			src = node_label_to_node_id[src]
			dst = node_label_to_node_id[dst]
			G.add_edge(src, dst)
	print "G.number_of_nodes()", G.number_of_nodes()
	print "G.number_of_edges()", G.number_of_edges()
	return G, node_label_to_node_id

def get_labels(label_file_path, node_label_to_node_id):
	node_id_to_label = {}
	label_name_to_label_id = {}
	current_open_label_id = 0
	with open(label_file_path) as f_label:
		next(f_label)
		next(f_label)
		for line in f_label:
			split_line = line.split("\t")
			node_id = int(split_line[0])
			node_id = node_label_to_node_id[node_id]
			label_id = int(split_line[1].split("label=")[1])
			if (not label_id in label_name_to_label_id):
				label_name_to_label_id[label_id] = current_open_label_id
				current_open_label_id += 1
			label_id = label_name_to_label_id[label_id]
			node_id_to_label[node_id] = label_id
	return node_id_to_label

def main():
	graph_file_path = "./Pubmed-Diabetes.DIRECTED.cites.tab"
	label_file_path = "./Pubmed-Diabetes.NODE.paper.tab"
	G, node_label_to_node_id = get_graph(graph_file_path)
	node_id_to_label = get_labels(label_file_path, node_label_to_node_id)
	print set(node_id_to_label.values())
	nx.write_edgelist(G, "../graph/graph.csv", delimiter=",", data=False)
	with open("../labels/labels", "w") as f_labels:
		for node_id, label in node_id_to_label.iteritems():
			f_labels.write(str(node_id) + "," + str(label) + "\n")

if __name__ == '__main__':
	main()