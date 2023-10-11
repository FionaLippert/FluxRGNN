import os
import os.path as osp
import pandas as pd
#import ruamel.yaml
import numpy as np
import argparse
from shutil import copy

parser = argparse.ArgumentParser()
parser.add_argument('hp_tuning_dir', type=str, help='directory with sub-directories that contain the output '
                                                    'of runs with different hyperparameter settings')
args = parser.parse_args()


def determine_best_hp():
    job_dirs = [f.path for f in os.scandir(args.hp_tuning_dir) if f.is_dir()]
    best_loss = np.inf
    for dir in job_dirs:
        # load cv summary
        df = pd.read_csv(osp.join(dir, 'summary.csv'))
        loss = df.final_val_loss.mean()

        if loss < best_loss:
            # copy config file to parent directory
            copy(osp.join(dir, 'config.yaml'), osp.dirname(args.hp_tuning_dir))
            best_loss = loss



if __name__ == "__main__":
    determine_best_hp()
