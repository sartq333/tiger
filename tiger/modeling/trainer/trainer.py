import copy
import os

import torch

from modeling.trainer import MetricCallback, InferenceCallback
from modeling.utils import create_logger, TensorboardWriter, DEVICE

LOGGER = create_logger(name=__name__)


class Trainer:
    def __init__(
            self,
            experiment_name,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            model,
            optimizer,
            loss_function,
            ranking_metrics,
            epoch_cnt=None,
            step_cnt=None,
            best_metric=None,
            epochs_threshold=40,
            valid_step=256,
            eval_step=256,
            checkpoint_dir='../checkpoints',
            checkpoint_step=512,
            resume=True
    ):
        self._experiment_name = experiment_name
        self._train_dataloader = train_dataloader
        self._validation_dataloader = validation_dataloader
        self._eval_dataloader = eval_dataloader
        self._model = model
        self._optimizer = optimizer
        self._loss_function = loss_function
        self._epoch_cnt = epoch_cnt
        self._step_cnt = step_cnt
        self._best_metric = best_metric
        self._epochs_threshold = epochs_threshold
        self._ranking_metrics = ranking_metrics
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_step = checkpoint_step
        self._resume = resume
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        tensorboard_writer = TensorboardWriter(self._experiment_name)

        self._metric_callback = MetricCallback(tensorboard_writer=tensorboard_writer, on_step=1)

        self._validation_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name='validation',
            model=model,
            dataloader=validation_dataloader,
            on_step=valid_step,
            metrics=ranking_metrics,
            pred_prefix='predictions',
            labels_prefix='labels'
        )

        self._eval_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name='eval',
            model=model,
            dataloader=eval_dataloader,
            on_step=eval_step,
            metrics=ranking_metrics,
            pred_prefix='predictions',
            labels_prefix='labels'
        )

    @property
    def _best_path(self):
        return f'{self._checkpoint_dir}/{self._experiment_name}_best.pth'

    @property
    def _latest_path(self):
        return f'{self._checkpoint_dir}/{self._experiment_name}_latest.pth'

    def _save_rolling(self, step_num, epoch_num, best_checkpoint, best_epoch, current_metric):
        """Atomically overwrite the single rolling snapshot (constant disk usage:
        the temp file is renamed over the old one, so the previous snapshot is gone)."""
        state = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'step_num': step_num,
            'epoch_num': epoch_num,
            'best_checkpoint': best_checkpoint,
            'best_epoch': best_epoch,
            'current_metric': float(current_metric),  # plain float: keep snapshot weights_only-safe
        }
        tmp = self._latest_path + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, self._latest_path)  # atomic; deletes the previous snapshot
        LOGGER.debug(f'Rolling checkpoint saved @ step {step_num} -> {self._latest_path}')

    def _try_resume(self):
        """Return (step_num, epoch_num, current_metric, best_epoch, best_checkpoint)
        restored from the rolling snapshot, or None for a fresh start."""
        if not (self._resume and os.path.exists(self._latest_path)):
            return None
        # weights_only=False: this is our own trusted snapshot and it bundles
        # non-tensor bookkeeping (ints/floats), which the weights_only loader rejects.
        state = torch.load(self._latest_path, map_location=DEVICE, weights_only=False)
        self._model.load_state_dict(state['model'])
        self._optimizer.load_state_dict(state['optimizer'])
        LOGGER.debug(
            f"Resuming from {self._latest_path}: step {state['step_num']}, epoch {state['epoch_num']}, "
            f"best {self._best_metric}={state['current_metric']:.5f}")
        return (state['step_num'], state['epoch_num'], state['current_metric'],
                state['best_epoch'], state['best_checkpoint'])

    def train(self):
        step_num = 0
        epoch_num = 0
        current_metric = 0
        best_epoch = 0
        best_checkpoint = None

        resumed = self._try_resume()
        if resumed is not None:
            step_num, epoch_num, current_metric, best_epoch, best_checkpoint = resumed

        LOGGER.debug('Start training...')

        while (step_num < 200_000):
            if best_epoch + self._epochs_threshold < epoch_num:
                LOGGER.debug(
                    'There is no progress during {} epochs. Finish training'.format(self._epochs_threshold))
                break

            LOGGER.debug(f'Start epoch {epoch_num}')
            for batch in self._train_dataloader:
                self._model.train()

                # Move to device
                for key, values in batch.items():
                    batch[key] = values.to(DEVICE)

                # Forward step
                batch.update(self._model(batch))
                loss = self._loss_function(batch)

                # Backward step
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                # Callbacks
                validation_metrics = self._validation_callback(step_num)
                evaluation_metrics = self._eval_callback(step_num)

                # Log metrics
                self._metric_callback(key='loss', value=loss.item(), step_num=step_num, prefix='train')
                for key, value in validation_metrics.items():
                    self._metric_callback(key=key, value=value, step_num=step_num, prefix='validation')
                for key, value in evaluation_metrics.items():
                    self._metric_callback(key=key, value=value, step_num=step_num, prefix='eval')

                # Update best checkpoint
                if self._best_metric is None:  # If no best metric is provided last checkpoint is taken
                    best_checkpoint = copy.deepcopy(self._model.state_dict())
                    best_epoch = epoch_num
                    assert False
                elif (
                    best_checkpoint is None  # If no best checkpoint exists this one is taken
                    or self._best_metric in validation_metrics and current_metric <= validation_metrics[self._best_metric]  # or if metrics improved compared to previous one
                ):
                    print(step_num, validation_metrics[self._best_metric], current_metric, evaluation_metrics.get(self._best_metric, float('nan')))
                    current_metric = validation_metrics[self._best_metric]
                    best_checkpoint = copy.deepcopy(self._model.state_dict())
                    best_epoch = epoch_num
                    # Persist best-so-far to disk so it survives an interrupted run
                    torch.save(best_checkpoint, self._best_path)
                    LOGGER.debug(f'New best {self._best_metric}={current_metric:.5f} @ step {step_num} -> {self._best_path}')

                # Rolling resumable snapshot at regular step intervals (single file, overwritten)
                if self._checkpoint_step and step_num % self._checkpoint_step == 0:
                    self._save_rolling(step_num, epoch_num, best_checkpoint, best_epoch, current_metric)

                step_num += 1

            epoch_num += 1
        LOGGER.debug('Training procedure has been finished!')
        return best_checkpoint

    def eval(self):
        evaluation_metrics = self._eval_callback(0)
        for key, value in evaluation_metrics.items():
            print(key, value)

    def save(self):
        LOGGER.debug('Saving model...')
        checkpoint_path = f'{self._checkpoint_dir}/{self._experiment_name}_final_state.pth'
        torch.save(self._model.state_dict(), checkpoint_path)
        LOGGER.debug('Saved model as {}'.format(checkpoint_path))

    def load(self, checkpoint):
        self._model.load_state_dict(checkpoint)
