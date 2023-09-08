import pytorch_lightning as pl
class TrainingLogsCallback(pl.Callback):
    # Important info in:
    # pytorch_lightning/callbacks/callback.py
    # lightning_fabric/loggers/csv_logs.py
    # pytorch_lightning/trainer/trainer.py
    def __init__(self, train_logger):
        super().__init__()
        self.train_logger = train_logger

    def on_train_epoch_end(self, trainer, pl_module):
        # on_train_batch_end

        all_logs = trainer.callback_metrics  # Alternatively: trainer.logged_metrics

        # trainer.callback_metrics= {}
        # There are both, trainer and validation metrics in callback_metrics (and logged_metrics as well )
        train_logs = {key: value for key, value in all_logs.items() if 'train_' in key}
        self.train_logger.log_metrics(train_logs) # , epoch=trainer.current_epoch) #, step=trainer.current_epoch)
        self.train_logger.save()


    def on_train_end(self, trainer, pl_module):
        # self.train_logger.finalize()
        self.train_logger.save()


class ValidationLogsCallback(pl.Callback):
    def __init__(self, val_logger):
        super().__init__()
        self.val_logger = val_logger

    def on_validation_epoch_end(self, trainer, pl_module):
        all_logs = trainer.callback_metrics
        # trainer.callback_metrics = {}
        val_logs = {key: value for key, value in all_logs.items() if 'val_' in key}
        self.val_logger.log_metrics(val_logs) # , epoch=trainer.current_epoch)
        # self.val_logger.log_metrics(val_logs) #, step=trainer.current_epoch)
        self.val_logger.save()

    def on_validation_end(self, trainer, pl_module):
        # self.val_logger.finalize()
        self.val_logger.save()

class BaselineValidationLogsCallback(pl.Callback):
    def __init__(self, val_logger):
        super().__init__()
        self.val_logger = val_logger

    def on_validation_epoch_end(self, trainer, pl_module):
        all_logs = trainer.callback_metrics
        # trainer.callback_metrics = {}
        val_logs = {key: value for key, value in all_logs.items() if 'val_' in key}
        self.val_logger.log_metrics(val_logs) # , epoch=trainer.current_epoch)
        # self.val_logger.log_metrics(val_logs) #, step=trainer.current_epoch)
        self.val_logger.save()

    def on_validation_end(self, trainer, pl_module):
        # self.val_logger.finalize()
        self.val_logger.save()


class BaselineTrainingLogsCallback(pl.Callback):
    def __init__(self, val_logger):
        super().__init__()
        self.val_logger = val_logger

    def on_validation_epoch_end(self, trainer, pl_module):
        all_logs = trainer.callback_metrics
        # trainer.callback_metrics = {}
        val_logs = {key: value for key, value in all_logs.items() if 'val_' in key}
        self.val_logger.log_metrics(val_logs) # , epoch=trainer.current_epoch)
        # self.val_logger.log_metrics(val_logs) #, step=trainer.current_epoch)
        self.val_logger.save()

    def on_validation_end(self, trainer, pl_module):
        # self.val_logger.finalize()
        self.val_logger.save()


