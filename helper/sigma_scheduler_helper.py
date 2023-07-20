import torch
class NullModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

def create_scheduler_mapping(training_steps_per_epoch, epochs, init_sigma):
    model = NullModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_sigma)
    scheduler_smoothing = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 3 * 10e-6)

    scheduler_mapping = []
    for _ in range(epochs * training_steps_per_epoch):
        optimizer.step()
        scheduler_mapping.append(scheduler_smoothing.get_lr()[0])
        scheduler_smoothing.step()

    return scheduler_mapping