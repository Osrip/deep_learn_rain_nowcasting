import torch
import einops


def pre_process_target(target: torch.Tensor, linspace_binning_params, s_num_bins_crossentropy, **__) -> torch.Tensor:
    # Creating binned target
    linspace_binning_min, linspace_binning_max, linspace_binning = self._linspace_binning_params
    target_binned = img_one_hot(target, self.s_num_bins_crossentropy, linspace_binning)
    target_binned = einops.rearrange(target_binned, 'b w h c -> b c w h')
    return target_binned

def pre_process_input(input_sequence: torch.Tensor) -> torch.Tensor:
    nan_mask = torch.isnan(input_sequence)
    input_sequence[nan_mask] = 0
    return input_sequence