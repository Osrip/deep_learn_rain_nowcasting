import torch
import math

class NullModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)


def create_scheduler_mapping(training_steps_per_epoch, epochs, init_sigma, s_multiple_sigmas, **__):
    model = NullModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_sigma)
    # sigma_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 3 * 10e-6)

    sigma_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1-2*10e-6) # this spans ~ 3 orders of magnitude

    # sigma_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
    #                                                      start_factor=0.5,  # The number we multiply learning rate in the first epoch
    #                                                      total_iters=training_steps_per_epoch * epochs)
    # sigma_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                     T_max=training_steps_per_epoch * epochs,  # Maximum number of iterations.
    #                                     eta_min=0.1)  # Minimum learning rate.

    scheduler_mapping = []
    for _ in range(epochs * training_steps_per_epoch):
        optimizer.step()
        scheduler_mapping.append(sigma_scheduler.get_lr()[0])
        sigma_scheduler.step()

    return scheduler_mapping, sigma_scheduler


def bernstein_polynomial(i, n, x):
    """
    Compute the Bernstein polynomial value for given i, n, and x.

    Parameters:
        i (int): Index of the sigma for which to compute the weight.
        n (int): Total number of sigmas.
        x (float): Input value.

    Returns:
        float: Weight value for the given i and x.

    # Example usage:
    n = 4  # Total number of sigmas
    x = 0.5  # Example input value (you can adjust this)

    # Calculate weights for all sigmas
    weights = [bernstein_polynomial(i, n, x) for i in range(1, n + 1)]

    print("Weights for sigmas:", weights)
    """
    n -= 1
    binomial_coefficient = math.comb(n, i)
    bernstein_term = binomial_coefficient * (x ** i) * ((1 - x) ** (n - i))
    # Formula source: https://de.wikipedia.org/wiki/Bernsteinpolynom
    return bernstein_term


def linear_schedule_0_to_1(curr_epoch, s_max_epochs, **__):
    x = (curr_epoch) / (s_max_epochs)
    return x


if __name__ == '__main__':
    #  Test plot for bernstein polynomials
    import numpy as np
    import matplotlib.pyplot as plt

    total_epochs = 50
    sigmas = [2, 4, 8, 16]

    weights = []
    x_values = []
    for curr_epoch in range(total_epochs):

        x = linear_schedule_0_to_1(curr_epoch, total_epochs)
        x_values.append(x)
        weights_per_epoch = []
        for i, sigma in enumerate(sigmas):
            weights_per_epoch.append(bernstein_polynomial(i, len(sigmas), x))

        weights.append(weights_per_epoch)
    weights = np.array(weights)

    # Plotting
    plt.figure(figsize=(10, 6))

    for i, sigma in enumerate(sigmas):
        inverted_i = len(sigmas) - i - 1 # Invert i to start with large value in schedule of the largest sigma
        plt.plot(range(total_epochs), weights[:, inverted_i], label=f'Sigma={sigma}')

    plt.xlabel('Epochs')  # Changed x-axis label to 'Epochs'
    plt.ylabel('Weights')
    plt.title('Weights vs Epochs for Different Sigmas')
    plt.legend()
    plt.grid(True)

    # Display x values on the right-hand side of the plot
    ax = plt.gca() # stands for "get current axis." It retrieves the current Axes instance in the current figure
    ax2 = ax.twinx() # creates a new Axes instance that shares the same x-axis as the current Axes (ax), but with a different y-axis
    ax2.set_ylabel('x', color='tab:red')
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_ylim(ax.get_ylim())
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.plot(range(total_epochs), x_values, color='tab:red', linestyle='--')  # Modified x-axis data

    plt.show()










