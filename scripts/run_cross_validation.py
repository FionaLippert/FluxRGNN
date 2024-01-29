from subprocess import Popen
import numpy as np


seed = 0

test_radars = [10, 20]

proc = Popen(['sbatch', 'run_neural_nets.job', 'model=FluxRGNN', 'logger.project=nexrad_hexagons_CV', 'trainer.max_epochs=2', 'missing_data_threshold=1.0', 'model.lr=3e-5', 'model.dropout_p=0.0', 'dataloader.batch_size=64', 'model.force_zeros=true', 'model.n_hidden=64', 'model.increase_horizon_rate=0.1', f'test_radars={test_radars}'])

if proc.poll() is None:
    proc.wait()
