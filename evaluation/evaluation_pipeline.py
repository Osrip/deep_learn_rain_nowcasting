import random

import pytorch_lightning as pl
from torch.utils.data import Subset, DataLoader

from evaluation.checkpoint_to_prediction import ckpt_to_pred
from evaluation.eval_with_baseline import ckpt_quick_eval_with_baseline
from evaluation.eval_with_baseline_fss import FSSEvaluationCallback
from helper.checkpoint_handling import get_checkpoint_names, load_from_checkpoint
from helper.memory_logging import format_duration
from load_data_xarray import FilteredDatasetXr
from plotting.plotting_pipeline import plot_logs_pipeline
import time


def evaluation_pipeline(
        data_set_vars, ckpt_settings,
        plot_training_logs=False
):
    '''
    This Pipeline is executed right after training or in 'plotting_only' mode
    '''

    (train_data_loader, validation_data_loader,
     training_steps_per_epoch, validation_steps_per_epoch,
     train_time_keys, val_time_keys, test_time_keys,
     train_sample_coords, val_sample_coords,
     radolan_statistics_dict,
     linspace_binning_params,) = data_set_vars

    # --- Get model checkpoint ---

    save_dir = ckpt_settings['s_dirs']['save_dir']

    all_checkpoint_names = get_checkpoint_names(save_dir)

    # Only do prediction for last checkpoint
    # TODO LOADING 'last' not 'best'
    checkpoints_to_evaluate = []
    checkpoint_name_1 = [name for name in all_checkpoint_names if 'best' in name][0]
    checkpoints_to_evaluate.append(checkpoint_name_1)

    # checkpoint_name_2 = [name for name in all_checkpoint_names if 'last' in name][0]
    # checkpoints_to_evaluate.append(checkpoint_name_2)

    datasets = {
        # 'train': train_data_loader.dataset,
        'val': validation_data_loader.dataset,
    }

    for checkpoint_name in checkpoints_to_evaluate:
        for dataset_name, dataset in datasets.items():
            print(f"\n STARTING EVALUATION ON BASELINE AND MODEL \n ...")
            print(f'\n Checkpoint: {checkpoint_name}, Dataset: {dataset_name} \n')


            model = load_from_checkpoint(
                save_dir,
                checkpoint_name,

                ckpt_settings,
                **ckpt_settings,
            )

            # --- Plot logs ---
            if plot_training_logs:
                plot_logs_pipeline(
                    training_steps_per_epoch,
                    model,
                    ckpt_settings, **ckpt_settings
                )

            # --- Quick evaluation and comparison to baseline over data set ---
            step_start_time = time.time()

            ckpt_quick_eval_with_baseline(
                model,
                checkpoint_name,
                dataset,
                dataset_name,
                radolan_statistics_dict,
                linspace_binning_params,

                ckpt_settings,
                **ckpt_settings
            )
            print(f'\n DONE. Took {format_duration(time.time() - step_start_time)} \n')

        # TODO: No enabled atm
        # # --- Generate predictions that are saved to a zarr ---
        # print(f"\n STARTING PREDICTIONS AND SAVING TO ZARR \n ...")
        # step_start_time = time.time()
        # ckpt_to_pred(
        #     model,
        #     checkpoint_name,
        #     train_time_keys, val_time_keys, test_time_keys,
        #     radolan_statistics_dict,
        #     linspace_binning_params,
        #     max_num_frames_per_split=3,
        #
        #     splits_to_predict_on=['val'],
        #     ckp_settings=ckpt_settings,
        #     **ckpt_settings,
        # )
        # print(f'\n DONE. Took {format_duration(time.time() - step_start_time)} \n')


def ckpt_quick_eval_with_fss(
        model,
        checkpoint_name,
        dataset,
        dataset_name,
        radolan_statistics_dict,
        linspace_binning_params,
        scales=[1, 3, 5, 9, 17, 33],  # Example scales (neighborhood sizes)
        thresholds=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],  # Example thresholds in mm/h
        ckpt_settings=None,  # Make sure to pass the settings of the checkpoint
        **__,
):
    """
    Evaluate a model checkpoint using FSS at different scales and thresholds.

    This function is designed to work alongside the standard evaluation with
    EvaluateBaselineCallback, focusing only on FSS calculation.

    Args:
        model: Trained model to evaluate
        checkpoint_name: Name of the checkpoint
        dataset: Dataset to use for evaluation
        dataset_name: Name of the dataset (for logging)
        radolan_statistics_dict: Statistics for normalization
        linspace_binning_params: Binning parameters
        scales: List of spatial scales to evaluate
        thresholds: List of precipitation thresholds to evaluate
        ckpt_settings: Settings from the checkpoint
    """
    print(f'\n STARTING FSS EVALUATION \n ...')
    step_start_time = time.time()

    # The model should already be in 'baseline' mode from previous evaluation
    if model.mode != 'baseline':
        print('Setting model to baseline mode for FSS evaluation')
        model.set_mode(mode='baseline')

    print('Initialize Dataset in baseline mode')
    # We use the same dataset setup as in ckpt_quick_eval_with_baseline
    sample_coords = dataset.sample_coords

    # Dataset should already be properly initialized for baseline evaluation
    # Just double-check the mode
    if not hasattr(dataset, 'mode') or dataset.mode != 'baseline':
        dataset = FilteredDatasetXr(
            sample_coords,
            radolan_statistics_dict,
            mode='baseline',
            settings=ckpt_settings,
            data_into_ram=False,
            baseline_path=ckpt_settings['s_baseline_path'],
            baseline_variable_name=ckpt_settings['s_baseline_variable_name'],
            num_input_frames_baseline=ckpt_settings['s_num_input_frames_baseline'],
        )

    # If dataset is already a Subset (from previous evaluation), use it as is
    if not isinstance(dataset, Subset) and ckpt_settings['s_subsample_dataset_to_len'] is not None:
        if ckpt_settings['s_subsample_dataset_to_len'] < len(dataset):
            print(
                f'Randomly subsample Dataset for FSS from length {len(dataset)} to {ckpt_settings["s_subsample_dataset_to_len"]}')
            # Use the same random seed for consistent subsampling between evaluations
            random.seed(42)
            subset_indices = random.sample(range(len(dataset)), ckpt_settings['s_subsample_dataset_to_len'])
            dataset = Subset(dataset, subset_indices)

    print(f'Dataset length for FSS evaluation: {len(dataset)}')

    print('Initializing FSS Evaluation Dataloader')
    data_loader = DataLoader(
        dataset,
        shuffle=False,  # Keep order consistent with other evaluations
        batch_size=ckpt_settings['s_batch_size'],
        drop_last=True,
        num_workers=0,  # To avoid freezing issues
        pin_memory=False,
    )

    print('Initializing FSS Evaluation Callback')
    fss_callback = FSSEvaluationCallback(
        scales=scales,
        thresholds=thresholds,
        linspace_binning_params=linspace_binning_params,
        checkpoint_name=checkpoint_name,
        dataset_name=dataset_name,
        settings=ckpt_settings,
    )

    print('Initializing Trainer for FSS Evaluation')
    trainer = pl.Trainer(
        callbacks=[fss_callback],
    )

    print('Starting FSS evaluation with trainer.predict')
    trainer.predict(
        model=model,
        dataloaders=data_loader,
        return_predictions=False
    )

    print(f'\n FSS EVALUATION COMPLETE. Took {format_duration(time.time() - step_start_time)} \n')
