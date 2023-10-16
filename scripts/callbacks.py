from pytorch_lightning.callbacks import Callback
from matplotlib import pyplot as plt
import numpy as np


class PredictionCallback(Callback):

    def on_validation_batch_end(self, trainer, pl_module, output, batch, batch_idx):
        """Called when the validation batch ends."""

        predictions = output['y_hat']

        if batch_idx == 0:

            # predictions has shape [cells, time]
            #cell_idx = np.random.randint(0, predictions.size(0)) #, size=min(5, predictions.size(0))
            
            n_cells = predictions.size(0)
            n_examples = 5

            img = []

            indices = np.where(batch.y[:, pl_module.t_context:].sum(1).detach().numpy() > 0)[0]
            print(f'{len(indices)} cells with non-zero values found')

            n_cells = len(indices)

            for idx in np.linspace(0, n_cells-1, min(n_cells, n_examples)).astype(int):
                cell_idx = indices[idx]
                if batch.y[cell_idx, pl_module.t_context:].sum() > 0:
                
                    fig, ax = plt.subplots()
                    
                    ax.plot(range(pl_module.horizon), predictions[cell_idx, :].detach().numpy(), label='prediction')
                    if 'source' in output:
                        ax.plot(range(pl_module.horizon), output['source'][cell_idx].view(-1).detach().numpy(), label='source')

                    if 'sink' in output:
                        ax.plot(range(pl_module.horizon), output['sink'][cell_idx].view(-1).detach().numpy(), label='sink')

                    ax.plot(range(-pl_module.t_context, pl_module.horizon), batch.y[cell_idx, :].detach().numpy(), label='data')
                    ax.legend()

                    trainer.logger.log_image(key=f'val_prediction_{cell_idx}_{batch_idx}', images=[fig])
            
                    plt.close()
                else:
                    print(f'cell {cell_idx} has no non-zero values')
