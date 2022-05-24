import gzip
import os

import jsonlines
import numpy as np

from grid2topo.habitat_utils import convert_transmat_to_point_quaternion

if __name__ == "__main__":
    # directory = "/data1/rxr_dataset/rxr-data/pose_traces/rxr_train/"
    directory = "../dataset/rxr-data/pose_traces/rxr_train/"
    file_list = os.listdir(directory)

    train_gz_file = "../dataset/rxr-data/rxr_train_guide.jsonl.gz"
    # train_gz_file = "/data1/rxr_dataset/rxr-data/rxr_train_guide.jsonl.gz"
    # train_gz_file = "/data1/rxr_dataset/rxr-data/rxr_train_follower.jsonl.gz"

    # npz file
    npzfile = np.load(directory + "000253_guide_pose_trace.npz")
    data_headers = npzfile.files
    print(data_headers)
    input()
    trans_mat_list = npzfile["intrinsic_matrix"]
    for trans_mat in trans_mat_list:
        position, angle_quaternion = convert_transmat_to_point_quaternion(trans_mat)
        print(position)
        print(angle_quaternion)
        input()

    # Check English guide has multiple instructions in one seen
    jsonl_file = gzip.open(train_gz_file)
    reader = jsonlines.Reader(jsonl_file)
    scene = "1LXtFkjw3qL"
    # scene = "1pXnuDYAj8r"
    for obj in reader:
        if obj["scan"] == scene and (obj["language"] == "en-IN" or obj["language"] == "en-US"):
            instruction_id = obj["instruction_id"]
            print(instruction_id)
            print(obj["instruction"])
            input()

    jsonl_file = gzip.open(train_gz_file)
    reader = jsonlines.Reader(jsonl_file)
    len_eng_obj = 0
    scan_list = []
    eng_instruction_list_by_scan = []
    for obj in reader:
        if obj["language"] == "en-IN" or obj["language"] == "en-US":
            if obj["scan"] in scan_list:
                scan_idx = scan_list.index(obj["scan"])
                eng_instruction_list_by_scan[scan_idx].append(obj["instruction"])
            else:
                scan_list.append(obj["scan"])
                eng_instruction_list_by_scan.append([obj["instruction"]])
            len_eng_obj = len_eng_obj + 1
    print(len_eng_obj)
    input()
    print(len(scan_list))
    input()
    instruction_by_scan_shape = [len(a) for a in eng_instruction_list_by_scan]
    print(instruction_by_scan_shape)
    input()
    print(sum(instruction_by_scan_shape))

    jsonl_file = gzip.open(train_gz_file)
    reader = jsonlines.Reader(jsonl_file)
    instruction_list = []
    scan_list_for_idx = []
    for obj in reader:
        instruction_list.append(obj["instruction"])
        scan_list_for_idx.append(obj["scan"])
    idx = 3
    for instruction in instruction_list:
        for eng_instruction in eng_instruction_list_by_scan[idx]:
            if instruction == eng_instruction:
                print(scan_list_for_idx[instruction_list.index(eng_instruction)])
