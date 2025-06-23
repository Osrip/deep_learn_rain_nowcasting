import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader

from data_pre_processing import FilteredDatasetXr


def create_data_loaders(
        train_sample_coords, val_sample_coords,
        train_oversampling_weights, val_oversampling_weights,
        radolan_statistics_dict,
        settings,
        s_batch_size,
        s_num_workers_data_loader,
        s_oversample_validation,
        s_oversample_train,
        s_train_samples_per_epoch,
        s_val_samples_per_epoch,
        s_oversampling_enabled=True,  # Default to True for backward compatibility
        **__
):
    '''
    Creates data loaders for training and validation data
    '''

    train_data_set = FilteredDatasetXr(
        train_sample_coords,
        radolan_statistics_dict,
        mode='train',
        settings=settings,
    )

    val_data_set = FilteredDatasetXr(
        val_sample_coords,
        radolan_statistics_dict,
        mode='train',
        settings=settings,
    )

    if s_train_samples_per_epoch is not None:
        train_samples_per_epoch = s_train_samples_per_epoch
    else:
        train_samples_per_epoch = len(train_data_set)

    if s_val_samples_per_epoch is not None:
        val_samples_per_epoch = s_val_samples_per_epoch
    else:
        val_samples_per_epoch = len(val_data_set)

    # If oversampling is disabled globally, ignore s_oversample_train and s_oversample_validation settings
    if not s_oversampling_enabled:
        if s_oversample_train or s_oversample_validation:
            print(
                "NOTE: s_oversample_train and s_oversample_validation settings are ignored because s_oversampling_enabled is False.")
        use_train_oversampling = False
        use_val_oversampling = False
    else:
        use_train_oversampling = s_oversample_train
        use_val_oversampling = s_oversample_validation

        # Validate oversampling weights are available and have correct dimensions
        if train_oversampling_weights is None or val_oversampling_weights is None:
            print("WARNING: Oversampling is enabled but weights are None. Falling back to standard sampling.")
            use_train_oversampling = False
            use_val_oversampling = False
        else:
            if not len(train_oversampling_weights) == len(train_data_set) == len(train_sample_coords):
                raise ValueError('Length of oversampling weights does not match length of data set or sample coords')
            if not len(val_oversampling_weights) == len(val_data_set) == len(val_sample_coords):
                raise ValueError('Length of oversampling weights does not match length of data set or sample coords')

    # Create weighted samplers if needed
    if use_train_oversampling:
        train_weighted_random_sampler = WeightedRandomSampler(
            weights=np.power(train_oversampling_weights, 0.5),
            num_samples=train_samples_per_epoch,
            replacement=True
        )

    if use_val_oversampling:
        val_weighted_random_sampler = WeightedRandomSampler(
            weights=np.power(val_oversampling_weights, 0.5),
            num_samples=val_samples_per_epoch,
            replacement=True
        )

    # Train data loader
    if use_train_oversampling:
        train_data_loader = DataLoader(
            train_data_set,
            sampler=train_weighted_random_sampler,  # <-- OVERSAMPLING
            batch_size=s_batch_size,
            drop_last=False,
            num_workers=s_num_workers_data_loader,
            pin_memory=True
        )
    else:
        train_data_loader = DataLoader(
            train_data_set,
            batch_size=s_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=s_num_workers_data_loader,
            pin_memory=True
        )

    # Validation data loader
    if use_val_oversampling:
        validation_data_loader = DataLoader(
            val_data_set,
            sampler=val_weighted_random_sampler,  # <-- OVERSAMPLING
            batch_size=s_batch_size,
            drop_last=False,
            num_workers=s_num_workers_data_loader,
            pin_memory=True
        )
    else:
        validation_data_loader = DataLoader(
            val_data_set,
            batch_size=s_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=s_num_workers_data_loader,
            pin_memory=True
        )

    return train_data_loader, validation_data_loader, train_samples_per_epoch, val_samples_per_epoch
