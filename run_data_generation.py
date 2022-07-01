import argparse
import gzip
from os import listdir
from os.path import isfile, join

import jsonlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path")
    args, _ = parser.parse_known_args()
    output_path = args.output_path

    directory = "../dataset/rxr-data/pose_traces/rxr_train/"
    pose_file_list = [f for f in listdir(directory) if isfile(join(directory, f))]

    train_guide_file = "../dataset/rxr-data/rxr_train_guide.jsonl.gz"
    scene_directory = "../dataset/mp3d_habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/"

    eng_pose_file_list = []
    eng_scene_list = []

    for i, pose_file in enumerate(pose_file_list):
        print(i)
        instruction_id = int(pose_file[0:6])

        jsonl_file = gzip.open(train_guide_file)
        reader = jsonlines.Reader(jsonl_file)

        for obj in reader:
            if obj["instruction_id"] == instruction_id and (obj["language"] == "en-IN" or obj["language"] == "en-US"):
                eng_pose_file_list.append(pose_file)
                eng_scene_list.append(obj["scan"])
