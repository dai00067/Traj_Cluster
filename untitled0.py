# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:54:20 2019

@author: zoed0
"""         
import sys
import os
sys.path.append('C:/Users/zoed0/Desktop/UBiAi/fengkong')
import json
import pandas as pd
import traclus_impl
from traclus_impl.geometry import Point
from traclus_impl.distance_functions import perpendicular_distance, angular_distance
from traclus_impl.generic_dbscan import dbscan
from traclus_impl.traclus_dbscan import *
from traclus_impl.trajectory_partitioning import *
from traclus_impl.parameter_estimation import *

##########
#some extracted inside functions, leave them alone
            
#hook functions
def get_dump_partitioned_trajectories_hook(file_name):
    if not file_name:
        return None
    
    def func(partitioned_stage_output):
        dict_trajs = map(lambda traj_line_seg: traj_line_seg.line_segment.as_dict(), 
                         partitioned_stage_output)
        with open(file_name, 'w') as output:
            output.write(json.dumps(dict_trajs))
    return func

def get_dump_clusters_hook(file_name):
    if not file_name:
        return None
    
    def func(clusters):
        all_cluster_line_segs = []
        for clust in clusters:
            line_segs = clust.get_trajectory_line_segments()
            dict_output = map(lambda traj_line_seg: traj_line_seg.line_segment.as_dict(), 
                              line_segs)
            all_cluster_line_segs.append(dict_output)
            
        with open(file_name, 'w') as output:
            output.write(json.dumps(all_cluster_line_segs))
    return func

#spikes_function to remove spikes
def with_spikes_removed(trajectory):
    if len(trajectory) <= 2:
        return trajectory[:]
        
    spikes_removed = []
    spikes_removed.append(trajectory[0])
    cur_index = 1
    while cur_index < len(trajectory) - 1:
        if trajectory[cur_index - 1].distance_to(trajectory[cur_index + 1]) > 0.0:
            spikes_removed.append(trajectory[cur_index])
        cur_index += 1
        
    spikes_removed.append(trajectory[cur_index])
    return spikes_removed
           
#functions for step 2
def filter_by_indices(good_indices, vals):
    vals_iter = iter(vals)
    good_indices_iter = iter(good_indices)
    out_vals = []
    
    num_vals = 0
    for i in good_indices_iter:
        if i != 0:
            raise ValueError("the first index should be 0, but it was " + str(i))
        else:
            for item in vals_iter:
                out_vals.append(item)
                break
            num_vals = 1
            break
            
    max_good_index = 0
    vals_cur_index = 1
    for i in good_indices_iter:
        max_good_index = i
        for item in vals_iter:
            num_vals += 1
            if vals_cur_index == i:
                vals_cur_index += 1
                out_vals.append(item)
                break
            else:
                vals_cur_index += 1
                
    for i in vals_iter:
        num_vals += 1
                
    if num_vals < 2:
        raise ValueError("list passed in is too short")
    if max_good_index != num_vals - 1:
        raise ValueError("last index is " + str(max_good_index) + \
                         " but there were " + str(num_vals) + " vals")
    return out_vals    

def consecutive_item_func_iterator_getter(consecutive_item_func, item_iterable):
    out_vals = []
    iterator = iter(item_iterable)
    last_item = None
    num_items = 0
    for item in iterator:
        num_items = 1
        last_item = item
        break
    if num_items == 0:
        raise ValueError("iterator doesn't have any values")
        
    for item in iterator:
        num_items += 1
        out_vals.append(consecutive_item_func(last_item, item))
        last_item = item
            
    if num_items < 2:
        raise ValueError("iterator didn't have at least two items")
        
    return out_vals



def dbscan_caller(cluster_candidates):
        line_seg_index = TrajectoryLineSegmentCandidateIndex(cluster_candidates, epsilon)
        return dbscan(cluster_candidates_index=line_seg_index, #TrajectoryLineSegmentCandidateIndex(cluster_candidates), \
                      min_neighbors=min_neighbors, \
                      cluster_factory=TrajectoryClusterFactory())
#######################################
#start here!     
        
file_num = []
traj_dict = []    
path = 'C:/Users/zoed0/Desktop/UBiAi/fengkong/testfile/tests_all'  
dirlist = os.listdir(path)
for dir in dirlist:
    file2 = os.listdir(path + '/' + dir)
    os.chdir(path + '/' + dir)
    file_num.append(len(file2))
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

#read in trajs, hook files    
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

#initialize the parameters
point_iterable_list = trajs
epsilon=0.002898
min_neighbors=16
min_num_trajectories_in_cluster=1
min_vertical_lines=1
min_prev_dist=0.0001
partitioned_points_hook=partitioned_traj_hook
clusters_hook=clusters_hook

#clean the traj input
cleaned_input = [] 
for traj in map(lambda l: with_spikes_removed(l), point_iterable_list):
        cleaned_traj = []
        if len(traj) > 1:
            prev = traj[0]
            cleaned_traj.append(traj[0])
            for pt in traj[1:]:
                if prev.distance_to(pt) == 0.0:
                    print(str(pt) + '***')
                    traj.remove(pt)
            for pt in traj[1:]:
                if prev.distance_to(pt) > 0.0:
                    cleaned_traj.append(pt)
                    prev = pt           
            if len(cleaned_traj) > 1:
                cleaned_input.append(cleaned_traj)
                     
        
###########################################################################
#Let's have fun!

#Steps of generating LS from trajs:
#step 1
#create a empty traj_line_seg_factory
trajectory_line_segment_factory = TrajectoryLineSegmentFactory()

#step 2
traj_line_segs = []
cur_trajectory_id = 0
for traj in cleaned_input:
    
    trajectory_id = cur_trajectory_id
    # good_indices is the returned partition_points
    good_indices = call_partition_trajectory(traj)
    
    # use partition_points to cut up traj
    good_point_iterable = filter_by_indices(good_indices,traj)
    # convert (par_point[i],par_point[i+1]) into LineSegment object
    line_segs = consecutive_item_func_iterator_getter(get_line_segment_from_points,item_iterable = good_point_iterable) 
    # one id generate(return) one TrajectoryLineSegment
    def create_traj_line_seg(line_seg):
        return trajectory_line_segment_factory.new_trajectory_line_seg(
                line_seg,\
                trajectory_id=trajectory_id)
    line_segments = map(create_traj_line_seg, line_segs)
    temp = 0
    for traj_seg in line_segments:
        traj_line_segs.append(traj_seg)
        temp += 1
    if temp <= 0:
        raise Exception()
    print(temp)
    cur_trajectory_id += 1   
#traj_line_segs are a list of TrajLineSegment for all trajs

#step3
#use TrajLineSegment to run DBSCAN, return clusters as a list
line_seg_index = TrajectoryLineSegmentCandidateIndex(traj_line_segs, epsilon)
clusters = dbscan(cluster_candidates_index = line_seg_index,  \
                  min_neighbors = min_neighbors, \
                  cluster_factory = TrajectoryClusterFactory())

#write out clusters
if clusters_hook:
            clusters_hook(clusters)

#step4            
#in genereated clusters,trace all Line Segments back to their traj_id:            
all_cluster_line_segs = []
traj_all_id = []
for clust in clusters:
            line_segs = clust.get_trajectory_line_segments()
            traj_id = map(lambda traj_line_seg: traj_line_seg.trajectory_id, 
                              line_segs)
            dict_output = map(lambda traj_line_seg: traj_line_seg.line_segment.as_dict(), 
                              line_segs)
            all_cluster_line_segs.append(dict_output)  
            traj_all_id.append(traj_id)        

#step6   
#organize traj and cluster id    
traj_a = []
cur_num = 0
for list in traj_all_id:
    for item in list:
        traj_a.append([cur_num,item])
    cur_num += 1    
traj_df = pd.DataFrame(traj_a,columns=['cluster_id','traj_id'])
#convert traj_id and cluster_id into pivot table
traj_df['count'] = 0
traj_pivot = traj_df.groupby(['cluster_id','traj_id']).agg('count')
traj_pivot = traj_pivot.reset_index()
#combine user_id
traj_pivot.traj_id = pd.to_numeric(traj_pivot.traj_id,errors='coerce')
def recur(n):
    return reduce(lambda x,y:x+y,file_num[:n])
for i in range(len(file_num)):
    if (i == 0):
        traj_pivot.loc[traj_pivot.traj_id < file_num[0],'user_id'] = int(i)
    else:
        traj_pivot.loc[(traj_pivot.traj_id>=recur(i)) & \
                       (traj_pivot.traj_id<recur(i+1)),'user_id'] = int(i)

############################         
#step5
#tuning parameters       
#adjust epsilon
#parsed_patition = parsed_patition[:1000]   #here to reduce calculated time, slice the patition
input_trajectories = []   
for traj in traj_line_segs:
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
    
    
    
    
    