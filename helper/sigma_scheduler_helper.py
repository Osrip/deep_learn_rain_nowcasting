import torch
class NullModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

def create_scheduler_mapping(training_steps_per_epoch, epochs, init_sigma):
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