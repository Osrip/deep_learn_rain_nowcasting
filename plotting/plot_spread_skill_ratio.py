import torch
import matplotlib.pyplot as plt
from einops import rearrange
from network_lightning import

def plot_spread_skill(model,
                      data_loader,
                      filter_and_normalization_params,
                      linspace_binning_params,
                      ps_device,
                      **__):
    with torch.no_grad():
        # Initialize an empty tensor for storing predictions
        predictions = torch.Tensor().to(ps_device)
        targets = torch.Tensor().to(ps_device)


        for i, (input_sequence, target) in enumerate(data_loader):
            input_sequence = input_sequence.to(ps_device)
            target = target.to(ps_device)

            model = model.to(ps_device)
            pred = model(input_sequence)

            # Concatenate the current batch predictions to the existing predictions tensor
            predictions = torch.cat((predictions, pred), dim=0)
            targets = torch.cat((targets, target), dim=0)

        predictions = rearrange(predictions, 'b c h w -> c (b h w)')
        targets = rearrange(targets, 'b h w -> (b h w)')

        argmaxed_predictions = torch.argmax(predictions, dim=0)
        argmaxed_targets = torch.argmax(targets, dim=0)
        class_distance = torch.abs(argmaxed_targets - argmaxed_predictions)
        # TODO: Mapping from bins to mm! Also look at whether one_hot_to_normed_mm considers cut-off at last bin!
        maxed_predictions = torch.max(predictions, dim=0).values
        maxed_predictions = torch.nn.Softmax()(maxed_predictions)
        # argmaxed targets - argmaxed predictions converted to mm is error
        # argmaxed targets color coding (which bin)
        # maxed predicitons certainty


        class_distance_np = class_distance.cpu().numpy()
        maxed_predictions_np = maxed_predictions.cpu().numpy()
        plt.figure()
        plt.scatter(class_distance_np, maxed_predictions_np)
        plt.show()




