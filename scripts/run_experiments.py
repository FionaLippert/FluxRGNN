from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import itertools as it
import os.path as osp
import os
from subprocess import Popen, PIPE
from datetime import datetime
import numpy as np
import pandas as pd
from shutil import copy
import re
#import ruamel.yaml

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    if cfg.verbose: print(f'hydra working directory: {os.getcwd()}')

    overrides = HydraConfig.get().overrides.task
    overrides = [o for o in overrides if (not "task" in o and 
                                          not "model=" in o and 
                                          not "datasource=" in o and
                                          not "model_dir=" in o)]
    overrides = " ".join(overrides)

    target_dir = osp.join(cfg.device.root, cfg.output_dir, cfg.datasource.name, cfg.model.name)
    print(f'target_dir = {target_dir}')
    os.makedirs(target_dir, exist_ok=True)

    if cfg.datasource.test_year == 'all':
        test_years = cfg.datasource.years
    else:
        test_years = [cfg.datasource.test_year]

    if cfg.task.name == 'hp_search':
        hp_grid_search(cfg, target_dir, test_years)
    elif cfg.task.name == 'train_eval' or cfg.task.name == 'train':
        train_eval(cfg, target_dir, test_years, overrides)
    elif cfg.task.name == 'eval':
        eval(cfg, target_dir, test_years, overrides)

def hp_grid_search(cfg: DictConfig, target_dir, test_years, timeout=10):

    hp_file, n_comb = generate_hp_file(cfg, target_dir)

    for year in test_years:
        # run inner cv for all hyperparameter settings
        output_dir = cfg.get('experiment', 'hp_grid_search')
        output_path = osp.join(target_dir, f'test_{year}', output_dir)

        if cfg.verbose: print(f"Start grid search for year {year}")

        # directory created by hydra, containing current config
        # including settings overwritten from command line
        config_path = osp.join(os.getcwd(), '.hydra')

        # option for running only parts of grid search
        n_start = cfg.get('hp_start', 1)

        # run inner cross-validation loop for all different hyperparameter settings
        if cfg.device.slurm:
            job_file = osp.join(cfg.device.root, cfg.task.slurm_job)
            proc = Popen(['sbatch', f'--array={n_start}-{n_comb}', job_file, cfg.device.root, output_path, config_path,
                          hp_file, str(year)], stdout=PIPE, stderr=PIPE)
        else:
            job_file = osp.join(cfg.device.root, cfg.task.local_job)
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            os.environ['HYDRA_FULL_ERROR'] = '1'
            proc = Popen([job_file, cfg.device.root, output_path, config_path,
                          hp_file, str(year), str(n_start), str(n_comb)], stdout=PIPE, stderr=PIPE)

        stdout, stderr = proc.communicate()
        start_time = datetime.now()

        # wait until job has been submitted (at most 10s)
        while True:
            if stderr:
                print(stderr.decode("utf-8"))   # something went wrong
            if stdout:
                print(stdout.decode("utf-8"))   # successful job submission
                return
            if (datetime.now() - start_time).seconds > timeout:
                print(f'timeout after {timeout} seconds')
                return


def train_eval(cfg: DictConfig, target_dir, test_years, overrides='', timeout=10):
    os.environ['HYDRA_FULL_ERROR'] = '1'
    
    for year in test_years:
        # determine best hyperparameter setting
        base_dir = osp.join(target_dir, f'test_{year}')
        hp_search_dir = cfg.get('hp_search_dir', 'hp_grid_search')
        input_dir = osp.join(base_dir, hp_search_dir)
        if osp.isdir(input_dir):
            determine_best_hp(input_dir)

            best_hp_config = OmegaConf.load(osp.join(base_dir, 'config.yaml'))
            best_hp_config.task = cfg.task

            with open(osp.join(base_dir, 'config.yaml'), 'w') as f:
                OmegaConf.save(config=best_hp_config, f=f)

        else:
            print(f'Directory "{hp_search_dir}" not found. Use standard config for training.')
            os.makedirs(base_dir, exist_ok=True)
            print(f'base_dir = {base_dir}')

            with open(osp.join(base_dir, 'config.yaml'), 'w') as f:
                OmegaConf.save(config=cfg, f=f)

            overrides = re.sub('[+]', '', overrides)

        # use this setting and train on all data except for one year
        output_dir = cfg.get('experiment', 'final_evaluation')
        # remove all '+' in overrides string
        #overrides = re.sub('[+]', '', overrides)
        output_path = osp.join(target_dir, f'test_{year}', output_dir)

        if cfg.verbose:
            print(f"Start train/eval for year {year}")
            print(f"Use overrides: {overrides}")
            
        config_path = osp.dirname(output_path)
        print(f'config_path = {config_path}')
        repeats = cfg.task.repeats
        if hasattr(cfg, 'trial'):
            array = cfg.trial
        else:
            array = f'1-{repeats}'

        if cfg.device.slurm:
            job_file = osp.join(cfg.device.root, cfg.task.slurm_job)
            gres = 1 if cfg.device.cuda else 0
            proc = Popen(['sbatch', f'--array={array}', f'--gres=gpu:{gres}', job_file, cfg.device.root, output_path, config_path,
                          str(year), overrides], stdout=PIPE, stderr=PIPE)
        else:
            job_file = osp.join(cfg.device.root, cfg.task.local_job)
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            os.environ['HYDRA_FULL_ERROR'] = '1'
            proc = Popen([job_file, cfg.device.root, output_path, config_path,
                          str(year), str(repeats)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        start_time = datetime.now()

        while True:
            if stderr:
                print(stderr.decode("utf-8"))
                return
            if stdout:
                print(stdout.decode("utf-8"))
                return
            if (datetime.now() - start_time).seconds > timeout:
                print(f'timeout after {timeout} seconds')
                return


def eval(cfg: DictConfig, target_dir, test_years, overrides='', timeout=10):
    os.environ['HYDRA_FULL_ERROR'] = '1'


    for year in test_years:
        assert hasattr(cfg, 'model_dir')

        cfg.model_dir = osp.join(cfg.device.root, cfg.model_dir)
        base_dir = osp.dirname(cfg.model_dir)
        output_path = cfg.model_dir

        cfg.sub_dir = ''

        print(f'model dir: {cfg.model_dir}')

        with open(osp.join(base_dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(config=cfg, f=f)

        overrides = re.sub('[+]', '', overrides)

        if cfg.verbose:
            print(f"Eval for year {year}")
            print(f"Use overrides: {overrides}")

        config_path = base_dir
        print(f'config_path = {config_path}')

        if cfg.device.slurm:
            job_file = osp.join(cfg.device.root, cfg.task.slurm_job)
            proc = Popen(['sbatch', job_file, cfg.device.root, output_path, config_path,
                          str(year), overrides], stdout=PIPE, stderr=PIPE)
        else:
            job_file = osp.join(cfg.device.root, cfg.task.local_job)
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            os.environ['HYDRA_FULL_ERROR'] = '1'
            proc = Popen([job_file, cfg.device.root, output_path, config_path,
                          str(year)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        start_time = datetime.now()

        while True:
            if stderr:
                print(stderr.decode("utf-8"))
                return
            if stdout:
                print(stdout.decode("utf-8"))
                return
            if (datetime.now() - start_time).seconds > timeout:
                print(f'timeout after {timeout} seconds')
                return


def determine_best_hp(input_dir: str):
    job_dirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    best_loss = np.inf
    for dir in job_dirs:
        # load cv summary
        df = pd.read_csv(osp.join(dir, 'summary.csv'))
        loss = np.nanmean(df.final_val_loss.values)

        if loss < best_loss:
            # copy config file to parent directory
            copy(osp.join(dir, 'config.yaml'), osp.dirname(input_dir))
            best_loss = loss


def generate_hp_file(cfg: DictConfig, target_dir):
    search_space = {k: v for k, v in cfg.hp_search_space.items() if k in cfg.model.keys()}
    hp_file = osp.join(target_dir, 'hyperparameters.txt')

    names, values = zip(*search_space.items())
    all_combinations = [dict(zip(names, v)) for v in it.product(*values)]

    with open(hp_file, 'w') as f:
        for combi in all_combinations:
            hp_str = " ".join([f'model.{name}={val}' for name, val in combi.items()]) + "\n"
            f.write(hp_str)

    if cfg.verbose:
        print("successfully generated hyperparameter settings file")
        print(f"File path: {hp_file}")
        print(f"Number of combinations: {len(all_combinations)} \n")

    return hp_file, len(all_combinations)


if __name__ == '__main__':

    run()


