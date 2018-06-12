import argparse
from collections import defaultdict
import cPickle
import os
import sys
import time
import sys

import algo_factory
import label_propagation
import pandas as pd
import utils


def get_experiment_results(param_tup):
  G, num_classes, known_labels, cluster_distribution, holdout, nodes_cluster, node2features, n, args, special_params = param_tup
  return run(G, num_classes, known_labels, cluster_distribution, holdout, nodes_cluster, node2features, args, special_params)


def run(G, num_classes, known_labels, cluster_distribution, holdout, nodes_cluster, node2features, args, special_params):
  inital_seed_set = known_labels.copy()
  # Ignore for accuracy the seed set and the holdout
  nodes_to_ignore_when_calculating_accuracy = inital_seed_set.copy()
  nodes_to_ignore_when_calculating_accuracy.update(holdout)

  holdout_accuracy_base, holdout_accuracy, bootstrapped, base = [], [], [], []
  should_continue = True
  t1 = time.time()
  special_params["features"] = node2features

  algo = algo_factory.algo_factory(G, num_classes, known_labels, args, special_params)

  while should_continue:
    print len(known_labels), len(nodes_cluster)
    t_prev = time.time()
    algo.fit()

    bootstrapped_predictions = algo.bootstrapped_predict()
    base_predictions = algo.base_predict()

    bootstrapped_accuracy = utils.calcualte_accuracy(nodes_cluster, nodes_to_ignore_when_calculating_accuracy, \
                          bootstrapped_predictions)
    bootstrapped.append(bootstrapped_accuracy)
    base_accuracy = utils.calcualte_accuracy(nodes_cluster, nodes_to_ignore_when_calculating_accuracy, \
                            base_predictions)
    base.append(base_accuracy)
    print "bootstrapped_accuracy %.2f" % bootstrapped_accuracy
    print "base_accuracy %.2f " %  base_accuracy

    holdout_confidence_scores = {}
    holdout_confidence_scores_reg = {}
    for node_id in holdout:
      if (node_id in bootstrapped_predictions):
        holdout_confidence_scores[node_id] = bootstrapped_predictions[node_id]
      if (node_id in base_predictions):
        holdout_confidence_scores_reg[node_id] = base_predictions[node_id]
    holdout_accuracy_current = utils.calcualte_accuracy(holdout, inital_seed_set, \
                                holdout_confidence_scores)
    holdout_accuracy.append(holdout_accuracy_current)
    holdout_accuracy_current_reg_algo = utils.calcualte_accuracy(holdout, inital_seed_set, \
                                holdout_confidence_scores_reg)
    holdout_accuracy_base.append(holdout_accuracy_current_reg_algo)
    
    # Apply non linearity
    confidence_scores = algo.get_bootstrapped_confidence_scores()
    num_prediced_nodes = utils.make_high_confidence_predictions(known_labels, cluster_distribution, holdout, \
                                  confidence_scores, \
                                  num_classes, len(G.nodes()), args, nodes_cluster)
    if (num_prediced_nodes < 10):
      should_continue = False
    algo.add_labeled_nodes(known_labels)
  t2 = time.time()
  print "repetition duraion", t2 -t1
  return (bootstrapped, base, holdout_accuracy, holdout_accuracy_base)

def calc_experiment_params(args):
  G = utils.load_graph(args.graph)
  nodes_cluster = utils.load_labels(args.all_labels)
  known_labels = utils.load_labels(args.seed_set)
  holdout = utils.load_labels(args.holdout)
  node2features = None
  if (not args.features is None):
    node2features = cPickle.load(open(args.features))
  special_params = {}

  if (args.model == "norm_lp" or args.model == "feature_diffusion_norm_lp"):
    special_params["M"] = label_propagation.get_graph_normalized_laplacian(G)
  if (args.model == "lp" or args.model == 'feature_diffusion_lp'):
    special_params["M"] = label_propagation.get_graph_laplacian(G)

  num_classes = max([nodes_cluster[node_id] for node_id in nodes_cluster]) + 1
  n = max(G.nodes())
  cluster_distribution = defaultdict(int)
  cluster_count = defaultdict(int)

  for node_id in nodes_cluster:
    cluster_count[nodes_cluster[node_id]] += 1
  
  for cluster_id in cluster_count:
    cluster_distribution[cluster_id] = cluster_count[cluster_id] / float(len(nodes_cluster))
  
  parameters = []
  parameters.append((G, num_classes, known_labels, cluster_distribution, holdout, nodes_cluster, node2features, n, args, special_params))
  return parameters

def create_arguments():
    parser = argparse.ArgumentParser()
    # General args
    parser.add_argument("--out_dir", help="Output directory" , type = str, default="./results")
    parser.add_argument("--graph", help="path to graph csv file. Each row should contain two pair of nodes e.g. node_id1,node_id2" , type = str)
    parser.add_argument("--all_labels", help="path to labels csv file. Eacho row should contain the node_id,class_id" , type = str)
    parser.add_argument("--seed_set", help="Same as all_labels format but only the seed set" , type = str)
    parser.add_argument("--holdout", help="Same as all_labels format but only holdout set" , type = str)
    parser.add_argument("--features", help="Path to features file. Cpickle dict key:node_id value sparse feature vector" , type = str)
    parser.add_argument("--model", help="lp|norm_lp|feature_diffusion_lp|feature_diffusion_norm_lp" , type = str, default="norm_lp")
    parser.add_argument("--num_nodes_per_cluster", help="number of nodes to add to seed set for each cluster", type=int, default=20)
    # When type==norm_lp
    parser.add_argument("--return_prob", help="the return probability. This option is only valid when type == norm_lp" ,
      type = float, default=0.0)
    # Bootstrapping params
    parser.add_argument("--iterations", help="number of iterations between applying non linear function", type=int, default=30)
    parser.add_argument("--percent_to_predict", help="top X percent to predict in each bootstrapped iteration", type=int, default=3)
    return parser

def create_dir_or_exit(out_dir):
  if (os.path.exists(out_dir)):
    raise Exception("Out dir must not exists")
  else:
    os.mkdir(out_dir)

def main():
  parser = create_arguments()
  args = parser.parse_args()
  create_dir_or_exit(args.out_dir)
  result = defaultdict(list)
  parameters = calc_experiment_params(args)
  for rep_num, par in enumerate(parameters):
    bootstrapped, base, holdout_accuracy, holdout_accuracy_base = \
      get_experiment_results(par)
    for i, b, n_b, h, h_reg in zip(range(len(bootstrapped)), bootstrapped, base, holdout_accuracy, holdout_accuracy_base):
      result["boosted"].append(b)
      result["base"].append(n_b)
      result["step"].append(i)
      result["rep"].append(rep_num)
      result["holdout_accuracy_boosted"].append(h)
      result["holdout_accuracy_reg"].append(h_reg)
  df = pd.DataFrame.from_dict(result)
  df.to_csv(os.path.join(args.out_dir, "result.csv"))
  cPickle.dump(args, open(os.path.join(args.out_dir, "params.cpickle"), "w"))
  with open(os.path.join(args.out_dir, "params.txt"), "w") as f_params:
    f_params.write(str(args) + "\n")

if __name__ == "__main__":
  main()
