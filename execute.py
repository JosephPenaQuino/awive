#!/home/joseph/anaconda3/envs/imageProcessing/bin/python3
'''execute both methods'''
import os
import argparse
import json
import time


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'
STEPS = 5

def execute_method(method_name, station_name, video_identifier, folder_path):
    cmd = f'./{method_name}.py {station_name} {video_identifier} -p {folder_path}'
    t = time.process_time()
    ret = os.popen(cmd).read()
    elapsed_time = 1000 * (time.process_time() - t)
    ret = json.loads(ret)
    print(method_name, end='\t')
    for i in range(STEPS):
        print(ret[str(i)]['velocity'], end='\t')
    print(elapsed_time)


def main(station_name, video_identifier, folder_path):
    '''execute both methods'''
    execute_method('sti', station_name, video_identifier, folder_path)
    execute_method('otv', station_name, video_identifier, folder_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "station_name",
        help="Name of the station to be analyzed")
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file")
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the config folder',
        type=str,
        default=FOLDER_PATH)
    args = parser.parse_args()
    main(
        station_name=args.station_name,
        video_identifier=args.video_identifier,
        folder_path=args.path,
        )
