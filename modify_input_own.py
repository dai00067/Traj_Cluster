# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:37:19 2019

@author: zoed0
"""
import sys
import os
sys.path.append('C:/Users/zoed0/Desktop/UBiAi/yunyin')
import traclus_impl
import click
from geometry import Point
import json
from coordination import run_traclus
import os
import pandas as pd
import math
from geometry import Point
import json
from coordination import run_traclus
import os
import unittest
from parameter_estimation import TraclusSimulatedAnnealingState
from parameter_estimation import TraclusSimulatedAnnealer
from geometry import Point
from simanneal import Annealer
import copy
import random
from mutable_float import MutableFloat
from coordination import the_whole_enchilada
from main import *
from coordination import *
#change path
os.chdir('C:/Users/zoed0/Desktop/UBiAi/fengkong/testfile/test_txts')
files = os.listdir(os.getcwd())
user1_num = len(files)
#read files, delete sequential replicates and convert files into a json dictionary
traj_dict = []
for file in files:
    f = open(file,'r')
    fcsv = pd.read_csv(f,sep = ',',header = None,names = ['a','b','y','x','idx'])
    ct = fcsv['x'].count()
    fcsv['idx'] = list(range(ct))
    for i in range(ct):
        if ((fcsv.iloc[i,2]==fcsv.iloc[i-1,2]) | (fcsv.iloc[i,3]==fcsv.iloc[i-1,3])):
            fcsv.iloc[i,4] = fcsv.iloc[i-1,4]
    f_drop = fcsv.drop_duplicates(['idx'])
    f_clean = f_drop[['x','y']]
    print(f_clean)
    f_list = f_clean.to_dict(orient='records')
    f_json = json.dumps(f_list)
    traj_dict.append(f_json)

os.chdir('C:/Users/zoed0/Desktop/UBiAi/fengkong/testfile/test_txt2')
file2 = os.listdir(os.getcwd())
user2_num = len(file2)
for file in file2:
    f = open(file,'r')
    fcsv = pd.read_csv(f,sep = ',',header = None,names = ['a','b','y','x','idx'])
    ct = fcsv['x'].count()
    fcsv['idx'] = list(range(ct))
    for i in range(ct):
        if ((fcsv.iloc[i,2]==fcsv.iloc[i-1,2]) | (fcsv.iloc[i,3]==fcsv.iloc[i-1,3])):
            fcsv.iloc[i,4] = fcsv.iloc[i-1,4]
    f_drop = fcsv.drop_duplicates(['idx'])
    f_clean = f_drop[['x','y']]
    print(f_clean)
    f_list = f_clean.to_dict(orient='records')
    f_json = json.dumps(f_list)
    traj_dict.append(f_json)
    
#change direction
os.chdir(os.path.abspath(os.path.pardir))

#write out as an input.txt
traj_last = traj_dict.pop()
fileObject = open('sampleList.txt', 'w')
fileObject.write(str('{"trajectories":['))
for ip in traj_dict:
	fileObject.write(ip)
	fileObject.write(',')
fileObject.write(traj_last)
fileObject.write(str(']}'))
fileObject.close()

#Adjust coefficients epsilon and MinLns
#read in original txt and write out three txt
input_file = 'sampleList.txt'
partitioned_trajectories_output_file_name = 'partitioned_stage_output.txt'     
clusters_output_file_name = 'clusters_output.txt'
output_file = 'traclus_output.txt'
partitioned_traj_hook = \
    get_dump_partitioned_trajectories_hook(partitioned_trajectories_output_file_name)
        
clusters_hook = get_dump_clusters_hook(clusters_output_file_name)

parsed_input = None 
with open(input_file, 'r') as input_stream:
    parsed_input = json.loads(input_stream.read()) 
traject = parsed_input.values()[0]
trajs = map(lambda traj: map(lambda pt: Point(**pt), traj), traject)
result = run_traclus(point_iterable_list=trajs,
                      epsilon=0.002898,
                      min_neighbors=16,
                      min_num_trajectories_in_cluster=1,
                      min_vertical_lines=1,
                      min_prev_dist=0.0001,
                      partitioned_points_hook=partitioned_traj_hook,
                      clusters_hook=clusters_hook)
dict_result = map(lambda traj: map(lambda pt: pt.as_dict(), traj), result)
with open(get_correct_path_to_file(output_file), 'w') as output_stream:
        output_stream.write(json.dumps(dict_result))

############
#read in partition and cluster.json
input_cluster = 'clusters_output.txt'
input_patition = 'partitioned_stage_output.txt'

parsed_cluster = None
with open(input_cluster, 'r') as input_stream:
        parsed_cluster = json.loads(input_stream.read())
parsed_patition = None
with open(input_patition, 'r') as input_stream:
        parsed_patition = json.loads(input_stream.read())    

############################        
#adjust epsilon
#parsed_patition = parsed_patition[:1000]   #here to reduce calculated time, slice the patition
input_trajectories = []   
for traj in parsed_patition:
    input_linesegment = []
    start = traj.values()[0]
    end = traj.values()[1]
    start_p = Point(start.values()[0],start.values()[1])
    end_p = Point(end.values()[0],end.values()[1])
    input_linesegment = [start_p,end_p]
    input_trajectories.append(input_linesegment)

initial_state = TraclusSimulatedAnnealingState(input_trajectories=input_trajectories, \
                                                       epsilon=0.002898)
traclus_sim_anneal = TraclusSimulatedAnnealer(initial_state=initial_state, \
                                                      max_epsilon_step_change=0.1)
traclus_sim_anneal.updates = 0
traclus_sim_anneal.steps = 100
best_state, best_energy = traclus_sim_anneal.anneal()
epsil = best_state.epsilon

#adjust MinLns
count = 0
sum_n = 0
for clust in parsed_cluster:
    if len(clust) > 1:
        count += 1
        sum_n = sum_n + len(clust)
avg_n = sum_n/count

#use the best epsilon and best MinLns to rerun the main function, get final output.txt
result_out = run_traclus(point_iterable_list=trajs,
                      epsilon=epsil,
                      min_neighbors=avg_n+1,
                      min_num_trajectories_in_cluster=1,
                      min_vertical_lines=1,
                      min_prev_dist=0.0001,
                      partitioned_points_hook=partitioned_traj_hook,
                      clusters_hook=clusters_hook)
dict_result_out = map(lambda traj: map(lambda pt: pt.as_dict(), traj), result_out)
with open(get_correct_path_to_file(output_file), 'w') as output_stream:
        output_stream.write(json.dumps(dict_result_out))
