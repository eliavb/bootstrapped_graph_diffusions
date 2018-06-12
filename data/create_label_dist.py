import argparse
import glob
import os
import cPickle
import pandas
from collections import defaultdict as dd
import csv


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help = 'input directory ' , type = str)
    parser.add_argument('--out_file', help = 'output file ' , type = str)
    return parser

def get_labels_dist(f_name):
	dist = dd(int)
	total = 0
	with open(f_name, "r") as f:
		for line in f:
			split_line = line.split(",")
			node_id = int(split_line[0])
			cluster = int(split_line[1])
			dist[cluster] += 1
			total += 1
	return dist, total

def create_result_table(input_dir, out_file):
	experiments_summaries = []	
	for root, dirs, files in os.walk(input_dir):
		for dir_name in dirs:
			full_dir_path = os.path.join(root, dir_name)
			pattern = os.path.join(full_dir_path, "*")
			files_in_dir = glob.glob(pattern)
			result_file = None
			param_file = None
			for f_name in files_in_dir:
				if (f_name.endswith("labels") and os.path.isfile(f_name)):
					dist, total = get_labels_dist(f_name)
					result_file = f_name + "_dist"
					with open(result_file, "w") as f_result:
						for cluster in dist:
							f_result.write(str(cluster) + "," + str(dist[cluster]) + "\n")

def main():
	parser 	=  add_arguments()
	args   	=  parser.parse_args()
	create_result_table(args.input_dir, args.out_file)

if __name__ == '__main__':
	main()