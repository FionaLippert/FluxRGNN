from pytorch_lightning.callbacks import Callback
from matplotlib import pyplot as plt


class PredictionCallback(Callback):

    def on_validation_batch_end(self, trainer, pl_module, predictions, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        fig, ax = plt.subplots()

        # predictions has shape [batch, time, cell]
        fig.plot(predictions[0, :, 0].detch().numpy(), label='prediction')
        fig.plot(batch.y[0, pl_module.t_context:, 0], label='data')
        fig.legend()

        pl_module.log_image(key=f'train_prediction', images=[fig])