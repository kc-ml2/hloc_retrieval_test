import gzip
import os

import jsonlines
import numpy as np

from utils.habitat_utils import convert_transmat_to_point_quaternion

if __name__ == "__main__":
    # directory = "/data1/rxr_dataset/rxr-data/pose_traces/rxr_train/"
    directory = "../dataset/rxr-data/pose_traces/rxr_train/"
    file_list = os.listdir(directory)

    train_gz_file = "../dataset/rxr-data/rxr_train_guide.jsonl.gz"
    test_gz_file = "../dataset/rxr-data/rxr_test_standard_public_guide.jsonl.gz"
    val_seen_gz_file = "../dataset/rxr-data/rxr_val_seen_guide.jsonl.gz"
    val_unseen_gz_file = "../dataset/rxr-data/rxr_val_unseen_guide.jsonl.gz"
    # train_gz_file = "/data1/rxr_dataset/rxr-data/rxr_train_guide.jsonl.gz"

    # npz file
    # npzfile = np.load(directory + "000253_guide_pose_trace.npz")
    # npzfile = np.load(directory + "000253_follower_pose_trace.npz")
    npzfile = np.load(directory + "036331_follower_pose_trace.npz")
    data_headers = npzfile.files
    print(data_headers)
    input()
    # for header in data_headers:
    #     print(npzfile[header])
    #     print(np.shape(npzfile[header]))
    #     input()
    print(npzfile["time"])
    print(np.shape(npzfile["time"]))
    input()
    print(npzfile["audio_time"])
    print(np.shape(npzfile["audio_time"]))
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

    for obj in reader:
        if obj["instruction_id"] == 36331:
            instruction_id = obj["instruction_id"]
            print(obj["timed_instruction"])
            print(np.shape(obj["timed_instruction"]))
            input()

    scene = "1LXtFkjw3qL"
    # scene = "1pXnuDYAj8r"
    id_list = []

    for obj in reader:
        if obj["scan"] == scene and (obj["language"] == "en-IN" or obj["language"] == "en-US"):
            instruction_id = obj["instruction_id"]
            print(obj["instruction"])
            input()
            id_list.append(instruction_id)

    jsonl_file = gzip.open(train_gz_file)
    reader = jsonlines.Reader(jsonl_file)
    len_eng_obj = 0
    train_gz_file = []
    eng_instruction_list_by_scan = []
    for obj in reader:
        if obj["language"] == "en-IN" or obj["language"] == "en-US":
            if obj["scan"] in train_gz_file:
                scan_idx = train_gz_file.index(obj["scan"])
                eng_instruction_list_by_scan[scan_idx].append(obj["instruction"])
            else:
                train_gz_file.append(obj["scan"])
                eng_instruction_list_by_scan.append([obj["instruction"]])
            len_eng_obj = len_eng_obj + 1
            print(obj["instruction_id"])
            input()
    print(len_eng_obj)
    input()
    print(len(train_gz_file))
    input()
    print(train_gz_file)
    input()
    # instruction_by_scan_shape = [len(a) for a in eng_instruction_list_by_scan]
    # print(instruction_by_scan_shape)
    # input()
    # print(sum(instruction_by_scan_shape))

    jsonl_file = gzip.open(val_seen_gz_file)
    reader = jsonlines.Reader(jsonl_file)
    len_eng_obj = 0
    val_seen_scan_list = []
    eng_instruction_list_by_scan = []
    for obj in reader:
        if obj["language"] == "en-IN" or obj["language"] == "en-US":
            if obj["scan"] in val_seen_scan_list:
                scan_idx = val_seen_scan_list.index(obj["scan"])
                eng_instruction_list_by_scan[scan_idx].append(obj["instruction"])
            else:
                val_seen_scan_list.append(obj["scan"])
                eng_instruction_list_by_scan.append([obj["instruction"]])
            len_eng_obj = len_eng_obj + 1
    print(len_eng_obj)
    input()
    print(len(val_seen_scan_list))
    input()
    print(val_seen_scan_list)
    input()

    n = 0
    for scan in train_gz_file:
        if scan in val_seen_scan_list:
            n = n + 1
    print(n)
