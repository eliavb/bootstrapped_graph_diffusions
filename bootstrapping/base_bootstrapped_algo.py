
class BaseBootstrapped(object):
	def __init__(self, G, num_classes, seed_nodes_to_cluster, args, special_params):
		self.bootstrapped_predictor = None
		self.base_predictor = None

	def add_labeled_nodes(self, seed_nodes_to_cluster):
		self.bootstrapped_predictor.add_labeled_nodes(seed_nodes_to_cluster)

	def fit(self):
		self.bootstrapped_predictor.fit()
		self.base_predictor.fit()

	def get_bootstrapped_confidence_scores(self):
		return self.bootstrapped_predictor.get_confidence_scores()

	def get_base_confidence_scores(self):
		return self.base_predictor.get_confidence_scores()

	def bootstrapped_predict(self):
		return self.bootstrapped_predictor.predict()

	def base_predict(self):
		return self.base_predictor.predict()