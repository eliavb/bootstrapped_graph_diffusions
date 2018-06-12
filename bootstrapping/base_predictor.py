

class BasePredictor(object):
	def __init__(self, G, num_classes, seed_nodes_to_cluster, args):
		"""
		G - networkx graph.
		num_classes - number possible classes
		seed_nodes_to_cluster- dict: key=node_id -> value=class
		args - command line args
		"""
		pass

	def add_labeled_nodes(self, seed_nodes_to_cluster):
		"""
		Add seed nodes to the predictor.
		seed_nodes_to_cluster- dict: key=node_id -> value=class
		"""
		pass

	def fit(self):
		"""
		Train the predictor
		"""
		pass

	def get_confidence_scores(self):
		"""
		Returns a list of (score, node_id, class) for each node_id and class.
		"""
		pass

	def predict(self):
		"""
		Predict a class for each node
		Returns a dict key=node_id -> value=class
		"""
		pass