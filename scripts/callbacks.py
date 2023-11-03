from pytorch_lightning.callbacks import Callback, EarlyStopping #, early_stopping
import torch
from matplotlib import pyplot as plt
import numpy as np

from fluxrgnn import utils


class PredictionCallback(Callback):

    def on_validation_batch_end(self, trainer, pl_module, output, batch, batch_idx):
        """Called when the validation batch ends."""

        plot_predictions(5, trainer, pl_module, output, batch, batch_idx, 'val')

    def on_test_batch_end(self, trainer, pl_module, output, batch, batch_idx):
        """Called when the test batch ends."""

        plot_predictions(5, trainer, pl_module, output, batch, batch_idx, 'test')

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""

        for m, value_list in pl_module.test_results.items():
            values = torch.concat(value_list, dim=0).reshape(-1, pl_module.horizon)

            fig, ax = plt.subplots()
            mean = values.mean(0)
            std = values.std(0)
            ax.plot(range(0, pl_module.horizon), mean)
            ax.fill_between(range(0, pl_module.horizon), mean - std, mean + std, alpha=0.2)
            trainer.logger.log_image(key=m.replace('/', '_'), images=[fig])
            ax.set(xlabel='horizon', ylabel=m)
            plt.close()


class GradNormCallback(Callback):

    def on_after_backward(self, trainer, pl_module):
        total_norm, max_norm = gradient_norm(pl_module)
        pl_module.log("model/total_grad_norm", total_norm)
        pl_module.log("model/max_grad_norm", max_norm)

# class MyEarlyStoppingCallback(EarlyStopping):
#
#     def __init__(self, *args, **kwargs):
#         super(MyEarlyStoppingCallback, self).__init__(monitor=)


class DebuggingCallback(Callback):

    def on_train_epoch_end(self, trainer, pl_module):

        if hasattr(pl_module, 'encoder'):
            print('####### encoder params #######')
            for name, value in pl_module.encoder.state_dict().items():
                print(f'{name}: {value}')

        if hasattr(pl_module, 'dynamics'):
            print('####### dynamics params #######')
            for name, value in pl_module.dynamics.state_dict().items():
                print(f'{name}: {value}')


def gradient_norm(model):
    total_norm = 0.0
    max_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            max_norm = max(max_norm, param_norm)
    total_norm = total_norm ** 0.5

    return total_norm, max_norm

def plot_predictions(n_plots, trainer, pl_module, output, batch, batch_idx, prefix='val'):

    predictions = output['y_hat']

    if batch_idx == 0:

        indices = np.where(batch.y[:, pl_module.t_context:].sum(1).detach().numpy() > 0)[0]
        print(f'{len(indices)} cells with non-zero values found')

        n_cells = len(indices)

        for idx in np.linspace(0, n_cells - 1, min(n_cells, n_plots)).astype(int):
            cell_idx = indices[idx]

            fig, ax = plt.subplots()

            ax.plot(range(pl_module.horizon), predictions[cell_idx, :].detach().numpy(), label='prediction')
            if 'source' in output:
                ax.plot(range(pl_module.horizon), output['source'][cell_idx].view(-1).detach().numpy(),
                        label='source')

            if 'sink' in output:
                ax.plot(range(pl_module.horizon), output['sink'][cell_idx].view(-1).detach().numpy(),
                        label='sink')

            ax.plot(range(-pl_module.t_context, pl_module.horizon), batch.y[cell_idx, :].detach().numpy(),
                    label='data')
            ax.legend()

            mse = utils.MSE(predictions[cell_idx, :], batch.y[cell_idx, pl_module.t_context:],
                            torch.logical_not(batch.missing)[cell_idx, pl_module.t_context:])
            ax.set(title=f'MSE = {mse:.6f}')

            trainer.logger.log_image(key=f'{prefix}_prediction_{cell_idx}_{batch_idx}', images=[fig])

            plt.close()




