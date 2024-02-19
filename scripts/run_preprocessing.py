from birds import datasets
from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    years = cfg.datasource.years
    print('preprocess data for years', years)
    data_root = osp.join(cfg.device.root, 'data')
    raw_data_root = cfg.get('raw_data_dir', osp.join(data_root, 'raw'))
    print(f'loading raw data from {raw_data_root}')

    cfg['datasource']['bird_scale'] = 1

    for year in years:
        res_info = f'_res={cfg.datasource.h3_resolution}' if cfg.model.edge_type == 'hexagons' \
            else f'_ndummy={cfg.datasource.n_dummy_radars}'

        base_dir = osp.join(data_root, 'preprocessed',
                              f'{cfg.t_unit}_{cfg.model.edge_type}_{cfg.datasource.buffer}' + res_info,
                              cfg.datasource.name)

        target_dir = osp.join(base_dir, cfg.season, str(year))

        # load all features and organize them into dataframes
        print(f'year {year}: start preprocessing')
        os.makedirs(target_dir, exist_ok=True)
        print(f'process dynamic features? {cfg.process_dynamic}')
        datasets.prepare_features(target_dir, raw_data_root, str(year), cfg.datasource.name,
                             random_seed=cfg.seed, edge_type=cfg.model.edge_type,
                             landcover_data=osp.join(base_dir, 'land_cover_hist.csv')
                             **cfg.datasource, **cfg)

if __name__ == "__main__":
    run()
