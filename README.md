# Bootstrapped Graph Diffusions: Exposing the Power of Nonlinearity

This is a implementation of Bootstrapped Graph Diffusion for the task of Graph based semi supervised learning, as described in our paper:
 
Eliav Buchnik, Edith Cohen, [Bootstrapped Graph Diffusions:
Exposing the Power of Nonlinearity](https://arxiv.org/pdf/1703.02618.pdf) (Sigmetrics 2018)

## Project dependencies
* numpy
* pandas
* networkx

## Data

To train the model you will need:
* Graph
* Labeling of the nodes of the graph
* Optional : node features

For convience we provide some of the datasets we used in our work. All data is located in the data folder. 

In the example below, we run on a citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/. 
We have two sets of splits. The first set named 'benchmark' is based on the work of [Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, Revisiting Semi-Supervised Learning with Graph Embeddings, ICML 2016](https://github.com/kimiyoung/planetoid) with an additional split to holdout set and test set to avoid overfitting.
The second set named 'random_splits' is just random splits of the raw data sets we used for robustness.

## Run example

Example 1:
```bash
python run.py  --out_dir ~/res --graph ~/bootstrapped_graph_diffusions/data/citeseer/benchmark/graph/graph.csv --all_labels ~/bootstrapped_graph_diffusions/data/citeseer/benchmark/setA/all_labels  --seed_set ~/bootstrapped_graph_diffusions/data/citeseer/benchmark/setA/seed_set --holdout ~/bootstrapped_graph_diffusions/data/citeseer/benchmark/setA/holdout  --model norm_lp
```

Example 2:
```bash
python run.py  --out_dir ~/res --graph ~/bootstrapped_graph_diffusions/data/citeseer/citeseer_random_splits/graph/graph.csv --all_labels ~/bootstrapped_graph_diffusions/data/citeseer/citeseer_random_splits/set0/all_labels  --seed_set ~/bootstrapped_graph_diffusions/data/citeseer/citeseer_random_splits/set0/seed_set --holdout ~/bootstrapped_graph_diffusions/data/citeseer/citeseer_random_splits/set0/holdout  --model norm_lp 
```

## Models

In cases where node features are not available the following models are possible:
* `lp`: label propogation using the graph Laplacian
* `norm_lp`: normalized label propogation using the normalized graph Laplacian

If node features are present then the following models are also possible:

* `feature_diffusion_norm_lp`: feature diffusion using the graph normazlied Laplacian
* `feature_diffusion_lp`: feature diffusion using the graph Laplacian

Full details in the paper.

## Script parameters
```bash
python run.py --help
usage: run.py [-h] [--out_dir OUT_DIR] [--graph GRAPH]
              [--all_labels ALL_LABELS] [--seed_set SEED_SET]
              [--holdout HOLDOUT] [--features FEATURES] [--model MODEL]
              [--num_nodes_per_cluster NUM_NODES_PER_CLUSTER]
              [--return_prob RETURN_PROB] [--iterations ITERATIONS]
              [--percent_to_predict PERCENT_TO_PREDICT]

optional arguments:
  -h, --help            show this help message and exit
  --out_dir OUT_DIR     Output directory
  --graph GRAPH         path to graph csv file. Each row should contain two
                        pair of nodes e.g. node_id1,node_id2
  --all_labels ALL_LABELS
                        path to labels csv file. Eacho row should contain the
                        node_id,class_id
  --seed_set SEED_SET   Same as all_labels format but only the seed set
  --holdout HOLDOUT     Same as all_labels format but only holdout set
  --features FEATURES   Path to features file. Cpickle dict key:node_id value
                        sparse feature vector
  --model MODEL         lp|norm_lp|feature_diffusion_lp|feature_diffusion_norm
                        _lp
  --num_nodes_per_cluster NUM_NODES_PER_CLUSTER
                        number of nodes to add to seed set for each cluster
  --return_prob RETURN_PROB
                        the return probability. This option is only valid when
                        type == norm_lp
  --iterations ITERATIONS
                        number of iterations between applying non linear
                        function
  --percent_to_predict PERCENT_TO_PREDICT
                        top X percent to predict in each bootstrapped
                        iteration

```

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{
  title={Bootstrapped Graph Diffusions: Exposing the Power of Nonlinearity},
  author={Eliav Buchnik, Edith Cohen},
  booktitle={ACM SIGMETRICS / International Conference on Measurement and Modeling of Computer Systems},
  year={2018}
}
```
