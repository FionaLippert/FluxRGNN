from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import traceback
import run_NNs, run_GAM, run_GBT, run_HA


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    print(f'hydra working directory: {os.getcwd()}')
    out = osp.join(cfg.output_dir, cfg.get('sub_dir', ''))
    print(f'output directory: {out}')
    os.makedirs(out, exist_ok=True)

    log_file = osp.join(out, 'log.txt')
    print(f'log file: {osp.abspath(log_file)}')
    log = open(log_file, 'w+')

    print('created log. start running experiment', file=log)
    print(f'output directory: {out}')

    log.flush()
    try:
        if cfg.model.name == 'GBT':
            run_GBT.run(cfg, out, log)
        elif cfg.model.name == 'GAM':
            run_GAM.run(cfg, out, log)
        elif cfg.model.name == 'HA':
            run_HA.run(cfg, out, log)
        else:
            run_NNs.run(cfg, out, log)
    except Exception:
        print(f'Error occurred! See {osp.abspath(log_file)} for more details.')
        print(traceback.format_exc(), file=log)
    print('flush log')
    log.flush()
    log.close()
    print('done')

if __name__ == "__main__":
    run()
