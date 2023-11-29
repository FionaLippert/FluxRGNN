from pytorch_lightning.callbacks import Callback, EarlyStopping #, early_stopping
import torch
from matplotlib import pyplot as plt
import numpy as np

from fluxrgnn import utils


class PredictionCallback(Callback):

    #def on_validation_batch_end(self, trainer, pl_module, output, batch, batch_idx):
    #    """Called when the validation batch ends."""

    #    plot_predictions(5, trainer, pl_module, output, batch, batch_idx, 'val')

    # def on_test_batch_end(self, trainer, pl_module, output, batch, batch_idx):
    #     """Called when the test batch ends."""
    #     indices = [0, 35, 71, 106, 142]
    #
    #     all_predictions = pl_module.test_predictions
    #     all_gt = pl_module.test_gt
    #
    #     for i in indices:
    #         prediction = all_predictions[i].cpu().numpy()
    #         gt = all_gt[i].cpu().numpy()
    #         key = f'test_prediction_{i}'
    #         plot_predictions(trainer, pl_module.t_context, pl_module.horizon, prediction, gt, key,
    #                          source=None, sink=None, node_flux=None)

    def on_test_end(self, trainer, pl_module):
        """Called when the test epoch ends."""

        # plot evaluation metrics over time
        for m, values in pl_module.test_metrics.items():
            #values = torch.concat(value_list, dim=0).reshape(-1, pl_module.horizon)

            fig, ax = plt.subplots()
            mean = np.nanmean(values.cpu().numpy(), axis=0)
            #std = values.std(0).cpu().numpy()
            ax.plot(range(0, pl_module.horizon), mean)
            #ax.fill_between(range(0, pl_module.horizon), mean - std, mean + std, alpha=0.2)
            trainer.logger.log_image(key=m.replace('/', '_'), images=[fig])
            ax.set(xlabel='horizon', ylabel=m)
            plt.close()

        # plot a few example predictions
        seq_idx = 0
        indices = [0, 35, 71, 106, 142]
        all_predictions = pl_module.test_results['test/predictions']
        all_gt = pl_module.test_results['test/measurements']

        for i in indices:
            prediction = all_predictions[seq_idx, i].cpu().numpy()
            gt = all_gt[seq_idx, i].cpu().numpy()
            key = f'test_prediction_{i}_{seq_idx}'
            plot_predictions(trainer, pl_module.t_context, pl_module.horizon, prediction, gt, key,
                             source=None, sink=None, node_flux=None)


class GradNormCallback(Callback):

    def on_after_backward(self, trainer, pl_module):
        total_norm, max_norm = gradient_norm(pl_module)
        pl_module.log("model/total_grad_norm", total_norm)
        pl_module.log("model/max_grad_norm", max_norm)



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

# def plot_predictions(indices, trainer, pl_module, output, batch, batch_idx, prefix='val'):
#
#     predictions = output['y_hat']
#
#     if batch_idx == 0:
#
#         #indices = np.where(batch.y[:, pl_module.t_context:].sum(1).detach().cpu().numpy() > 0)[0]
#         #print(f'{len(indices)} cells with non-zero values found')
#
#         n_cells = len(indices)
#
#         for idx, cell_idx in enumerate(indices): #np.linspace(0, n_cells - 1, min(n_cells, n_plots)).astype(int):
#             #cell_idx = indices[idx]
#
#             fig, ax = plt.subplots(figsize=(6, 3))
#
#             ax.plot(range(pl_module.horizon), pl_module.to_raw(predictions[cell_idx, :]).detach().cpu().numpy(), label='prediction')
#             if 'source' in output:
#                 ax.plot(range(pl_module.horizon), output['source'][cell_idx].view(-1).detach().cpu().numpy(),
#                         label='source')
#
#             if 'sink' in output:
#                 ax.plot(range(pl_module.horizon), output['sink'][cell_idx].view(-1).detach().cpu().numpy(),
#                         label='sink')
#
#             ax.plot(range(-pl_module.t_context, pl_module.horizon), pl_module.to_raw(batch.y[cell_idx, :]).detach().cpu().numpy(),
#                     label='data')
#             ax.legend()
#
#             #mse = utils.MSE(predictions[cell_idx, :], batch.y[cell_idx, pl_module.t_context:],
#             #                torch.logical_not(batch.missing)[cell_idx, pl_module.t_context:])
#             #ax.set(title=f'MSE = {mse:.6f}')
#             ax.set(xlabel='forecasting horizon', ylabel='birds per km2')
#
#             trainer.logger.log_image(key=f'{prefix}_prediction_{cell_idx}_{batch_idx}', images=[fig])
#
#             plt.close()

def plot_predictions(trainer, context, horizon, prediction, gt, key,
                     source=None, sink=None, node_flux=None):

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(range(horizon), prediction, label='prediction')
    if source is not None:
        ax.plot(range(horizon), source, label='source')

    if sink is not None:
        ax.plot(range(horizon), sink, label='sink')

    ax.plot(range(-context, horizon), gt, label='data')
    ax.legend()

    ax.set(xlabel='forecasting horizon', ylabel='birds per km2')

    trainer.logger.log_image(key=key, images=[fig])

    plt.close()




