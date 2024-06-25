import torch

# TODO not finished as I decided to just give the pysteps fss another shot

def fss(
        target_one_hot: torch.Tensor,
        prediction_one_hot: torch.Tensor,
        threshold_bin: int,
        window_size: int
        ):

    '''
    This function calculates the FSS (Fractions Skill Score) on the binned targets and predictions.
    The threshold_bin variable counts the current pixel if the max bin of the current pixel is lower or equal to the
    threshold bin
    '''
    pass

def convert_to_count(
        one_hot_data: torch.Tensor,
        threshold_bin,
        ):
    # Initialize tensor without channel dim
    new_shape = one_hot_data.shape[0], one_hot_data.shape[2], one_hot_data.shape[3]
    count = torch.zeros(*new_shape)

    # Find max bins
    target_max_bin_indices = count.argmax(dim=1)

    # Where are the max bins smaller or equal to threshold bin?
    max_bin_lower_equal_thresold_boo = target_max_bin_indices <= threshold_bin

    count[one_hot_data]

