import torch
import matplotlib.pyplot as plt
from einops import rearrange
from helper.pre_process_target_input import pre_process_input, pre_process_target
import numpy as np

def plot_spread_skill(model,
                      data_loader,
                      filter_and_normalization_params,
                      linspace_binning_params,
                      checkpoint_name_no_ending,
                      settings,
                      ps_device,
                      ps_runs_path,
                      s_local_machine_mode,
                      **__):
    with torch.no_grad():
        # Initialize an empty tensor for storing predictions
        predictions = torch.Tensor().to(ps_device)
        targets = torch.Tensor().to(ps_device)

        for i, (input_sequence, target) in enumerate(data_loader):
            input_sequence = input_sequence.to(ps_device)
            target = target.to(ps_device)

            # TODO implement correct pre procesing in other chkpoint plots as well!
            input_sequence = pre_process_input(input_sequence)
            target = pre_process_target(target, linspace_binning_params, **settings)

            model = model.to(ps_device)
            pred = model(input_sequence)

            # Concatenate the current batch predictions to the existing predictions tensor
            predictions = torch.cat((predictions, pred), dim=0)
            targets = torch.cat((targets, target), dim=0)
            # Only use a certain amount of samples
            if i == 10:
                break

        predictions = rearrange(predictions, 'b c h w -> c (b h w)')
        targets = rearrange(targets, 'b c h w -> c (b h w)')

        argmaxed_predictions = torch.argmax(predictions, dim=0)
        argmaxed_targets = torch.argmax(targets, dim=0)
        softmaxed_predictions = torch.nn.Softmax(dim=0)(predictions)
        maxed_predictions = torch.max(softmaxed_predictions, dim=0).values

        class_distance = torch.abs(argmaxed_targets - argmaxed_predictions)

        # # Add random noise to the discrete class distance between -.05 and 0.5
        # class_distance = class_distance - torch.rand(class_distance.shape).to(ps_device) - .5

        # TODO: Mapping from bins to mm! Also look at whether one_hot_to_normed_mm considers cut-off at last bin!
        # maxed_predictions = torch.nn.Softmax()(maxed_predictions)
        # argmaxed targets - argmaxed predictions converted to mm is error
        # argmaxed targets color coding (which bin)
        # maxed predicitons certainty



        class_distance_np = class_distance.cpu().numpy()
        maxed_predictions_np = maxed_predictions.cpu().numpy()

        plt.figure()
        # plt.xscale('log')
        if s_local_machine_mode:
            plt.scatter(class_distance_np, maxed_predictions_np, s=0.3, alpha=0.2)
        else:
            # plt.scatter(class_distance_np, maxed_predictions_np, s=0.001, alpha=0.05)
            for curr_class_distance in range(max(class_distance_np)):
                maxed_predictions_of_curr_class_distance = maxed_predictions_np[class_distance_np == curr_class_distance]
                curr_class_distance_filled = np.full(
                    np.shape(maxed_predictions_of_curr_class_distance),
                    curr_class_distance)
                # Add random noise to the discrete class distance between -.05 and 0.5
                curr_class_distance_filled = curr_class_distance_filled - np.random.rand(len(curr_class_distance_filled)) - 0.5

                plt.scatter(
                    curr_class_distance_filled,
                    maxed_predictions_of_curr_class_distance,
                    s=0.001*curr_class_distance,
                    alpha=min(1, 0.05*curr_class_distance),
                    color='blue')
        plt.xlabel('Class distance (bins)')
        plt.ylabel('Certainty (max bin)')
        plt.show()

        save_path_name = f'{ps_runs_path}/plots/scatter_spread_skill_{checkpoint_name_no_ending}.png'
        plt.savefig(save_path_name, bbox_inches='tight', dpi=50)




