# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""Aggregates multiple tensorbaord runs"""

import ast
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event
import math
import copy 

FOLDER_NAME = 'aggregates'
class BaseEvent:
    def __init__(self, wall_time, value, step):
        self.wall_time = wall_time
        self.value = value
        self.step = step

    def __str__(self):
        return "BaseEvent(wall_time=" + str(self.wall_time) + ", step=" + str(self.step) + ", value=" + str(self.value) + ")"

def interpolate(timestep, current_index, timestep_size, dataset):
    if current_index + 1 >= len(dataset):
        return None, current_index
    
    step_skips = 0
    while int(current_index + step_skips + 1) < len(dataset) and dataset[int(current_index + step_skips + 1)].step < timestep:
        step_skips += 1
    #step_skips = (timestep - dataset[current_index].step)//skip_gap #
    #print(step_skips)
    current_index = int(current_index + step_skips)
    #print("showtime: 0", timestep, dataset[current_index].step, current_index, dataset[current_index - 1].step)
    if current_index + 1 >= len(dataset):
        return None, current_index
    
    x1 = dataset[current_index].step
    x2 = dataset[current_index+1].step
    data_gap = x2 - x1
    if data_gap == 0:
        x1 = dataset[current_index].step
        x2 = dataset[current_index+2].step
        data_gap = x2 - x1

    proportion = (timestep - x1)/data_gap
    if proportion < 0 or 1 - proportion < 0:
        print("ze bug", x1, x2, timestep, proportion)
        quit()
    wall_time = dataset[current_index].wall_time * (1 - proportion) + dataset[current_index + 1].wall_time * proportion
    value = dataset[current_index].value * (1 - proportion) + dataset[current_index + 1].value * proportion
    step = timestep
    newEvent = BaseEvent(wall_time, value, step)
    newEvent.index1 = current_index
    newEvent.index2 = current_index + 1

    return newEvent, current_index




def extract(dpath, subpath):
    scalar_accumulators = []
    directories = os.listdir(dpath)
    for dname in directories:
        if dname != FOLDER_NAME:
            subdirectory = os.listdir(dpath/dname)[0]
            scalar_accumulators.append(EventAccumulator(str(dpath / dname / subdirectory)).Reload().scalars)
    #scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]
    
    # Get and validate all scalar keys
    all_keys = {}
    for accumulator in scalar_accumulators:
        key_list = accumulator.Keys()
        for key in key_list:
            if key in all_keys:
                all_keys[key] += 1
            else:
                all_keys[key] = 1
        
    keys = []
    for key in all_keys:
        if all_keys[key] == len(scalar_accumulators):
            keys.append(key)
    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys]
    # Get and validate all steps per key
    all_steps_per_key = [[tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]
    step_data = []
    for i in range(len(all_scalar_events_per_key)):
        #print(len(all_scalar_events_per_key[i]), i, keys[i])
        max_val = -math.inf
        min_val = -math.inf
        max_length = -math.inf
        for j in range(len(all_scalar_events_per_key[i])):
            #print(len(all_scalar_events_per_key[i][j]), all_scalar_events_per_key[i][j][-1])
            if all_scalar_events_per_key[i][j][-1].step > max_val:
                max_val = all_scalar_events_per_key[i][j][-1].step
            if len(all_scalar_events_per_key[i][j]) > max_length:
                max_length = len(all_scalar_events_per_key[i][j])
            if all_scalar_events_per_key[i][j][0].step > min_val:
                min_val = all_scalar_events_per_key[i][j][0].step
        step_data.append([max_val, min_val, max_length, (max_val - min_val)/(0.000000001 + max_length)])
    interpolated_all_steps_per_key = []
    for i in range(len(all_scalar_events_per_key)):
        if step_data[i][2] <= 1:
            interpolated_all_steps_per_key.append(all_scalar_events_per_key[i])

        else:
            interpolated_all_steps_per_key.append([])
            current_scalar_step = []
            for event_id in range(len(all_scalar_events_per_key[i])):
                current_scalar_step.append(0)

            for event_id in range(len(all_scalar_events_per_key[i])):
                data = all_scalar_events_per_key[i][event_id]
                interpolated_all_steps_per_key[-1].append([])
                for current_time_index in range(step_data[i][2]):
                    current_timestep = step_data[i][1] + current_time_index * step_data[i][3]
                    interpolated_event, current_index = interpolate(current_timestep, current_scalar_step[event_id], step_data[i][3], all_scalar_events_per_key[i][event_id])
                    current_scalar_step[event_id] = current_index
                    if not interpolated_event is None:
                        interpolated_all_steps_per_key[-1][-1].append(interpolated_event)
                    else:
                        break
            #     for k in range(len(interpolated_all_steps_per_key[-1][-1])):
            #         print(k, interpolated_all_steps_per_key[-1][-1][k])        
            # quit()
    result_dict = {}
    for i in range(len(keys)):
        result_dict[keys[i]] = interpolated_all_steps_per_key[i]
    #interpolate this crap so it works .... 
    # for i, all_steps in enumerate(all_steps_per_key):
    #     assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
    #         keys[i], [len(steps) for steps in all_steps])

    # steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # # Get and average wall times per step per key
    # wall_times_per_key = [np.mean([tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events], axis=0)
    #                       for all_scalar_events in all_scalar_events_per_key]

    # # Get values per step per key
    # values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
    #                   for all_scalar_events in all_scalar_events_per_key]

    # all_per_key = dict(zip(keys, zip(steps_per_key, wall_times_per_key, values_per_key)))
    return result_dict


def aggregate_to_summary(dpath, aggregation_ops, extracts_per_subpath):
    for op in aggregation_ops:
        for subpath, all_per_key in extracts_per_subpath.items():
            path = dpath / FOLDER_NAME / op.__name__
            write_summary(path, all_per_key, op)
    # print(extracts_per_subpath)
    # print(aggregation_ops)
    # quit()
    # for op in aggregation_ops:
    #     for subpath, all_per_key in extracts_per_subpath.items():
    #         path = dpath / FOLDER_NAME / op.__name__ / dpath.name / subpath
    #         aggregations_per_key = {key: (steps, wall_times, op(values, axis=0)) for key, (steps, wall_times, values) in all_per_key.items()}
    #         write_summary(path, aggregations_per_key)


def write_summary(dpath, all_per_key, op):
    writer = tf.summary.FileWriter(dpath)
    scalar = tf.placeholder(tf.float32, shape=[])
    for key in all_per_key:
        #key name .... 
        #key_summary = tf.summary.scalar(name=key, tensor=scalar)
        data = all_per_key[key]
        max_iter = -1
        for event_id in range(len(data)):
            if max_iter < len(data[event_id]):
                max_iter = len(data[event_id])

        print("Writing", key, "operation", op.__name__)
        for step in range(max_iter):
            value_list = []
            step_list = []
            wall_time = []
            for event_id in range(len(data)):
                if step < len(data[event_id]):
                    value_list.append(data[event_id][step].value)
                    step_list.append(data[event_id][step].step)
                    wall_time.append(data[event_id][step].wall_time)
            new_value = op(value_list)
            new_step = int(np.mean(step_list))
            new_wall_time = np.mean(wall_time)
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=new_value)])
            scalar_event = Event(wall_time=new_wall_time, step=new_step, summary=summary)
            writer.add_event(scalar_event)

    # for key, (steps, wall_times, aggregations) in aggregations_per_key.items():
    #     for step, wall_time, aggregation in zip(steps, wall_times, aggregations):
    #         summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=aggregation)])
    #         scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
    #         writer.add_event(scalar_event)

    writer.flush()


def aggregate_to_csv(dpath, aggregation_ops, extracts_per_subpath):
    raise("Not yet implemented .... ")
    # for subpath, all_per_key in extracts_per_subpath.items():
    #     for key, (steps, wall_times, values) in all_per_key.items():
    #         aggregations = [op(values, axis=0) for op in aggregation_ops]
    #         write_csv(dpath, subpath, key, dpath.name, aggregations, steps, aggregation_ops)


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def write_csv(dpath, subpath, key, fname, aggregations, steps, aggregation_ops):
    path = dpath / FOLDER_NAME

    if not path.exists():
        os.makedirs(path)

    file_name = get_valid_filename(key) + '-' + get_valid_filename(subpath) + '-' + fname + '.csv'
    aggregation_ops_names = [aggregation_op.__name__ for aggregation_op in aggregation_ops]
    df = pd.DataFrame(np.transpose(aggregations), index=steps, columns=aggregation_ops_names)
    df.to_csv(path / file_name, sep=';')


def aggregate(dpath, output, subpaths):
    name = dpath.name

    aggregation_ops = [np.mean, np.min, np.max, np.median, np.std, np.var]

    ops = {
        'summary': aggregate_to_summary,
        'csv': aggregate_to_csv
    }

    print("Started aggregation {}".format(name))

    extracts_per_subpath = {subpath: extract(dpath, subpath) for subpath in subpaths}
    ops.get(output)(dpath, aggregation_ops, extracts_per_subpath)

    print("Ended aggregation {}".format(name))


if __name__ == '__main__':
    def param_list(param):
        p_list = ast.literal_eval(param)
        if type(p_list) is not list:
            raise argparse.ArgumentTypeError("Parameter {} is not a list".format(param))
        return p_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="main path for tensorboard files", default=os.getcwd())
    parser.add_argument("--subpaths", type=param_list, help="subpath sturctures", default=[''])
    parser.add_argument("--output", type=str, help="aggregation can be saves as tensorboard file (summary) or as table (csv)", default='summary')

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(path))

    # subpaths = [path / dname / subpath for subpath in args.subpaths for dname in os.listdir(path) if dname != FOLDER_NAME]

    # for subpath in subpaths:
    #     if not os.path.exists(subpath):
    #         raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(subpath))

    # if args.output not in ['summary', 'csv']:
    #     raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))
    aggregate(path, args.output, args.subpaths)
