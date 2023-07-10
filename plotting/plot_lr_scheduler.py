import matplotlib.pyplot as plt
import torch
import json


class NullModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

def plot_lr_schedule(lr_scheduler, training_steps_per_epoch, epochs, ps_sim_name, **__):
    model = NullModule()
    optimizer = torch.optim.Adam(model.parameters())



    lrs = []
    for _ in range(epochs * training_steps_per_epoch):
        optimizer.step()
        lrs.append(lr_scheduler.get_lr())
        lr_scheduler.step()

    plt.plot(lrs)
    save_name = 'lr_scheduler'
    save_path_name = 'runs/{}/plots/{}'.format(ps_sim_name, save_name)
    plt.ylabel('Learning Rate')
    plt.xlabel('Step (total corresponds to steps in training)')
    plt.title('LR scheduler: {}'.format(lr_scheduler.__class__.__name__))
    plt.savefig(save_path_name, dpi=200, bbox_inches='tight')
    plt.show()

    # Save hyperparams to txt
    scheduler_state = lr_scheduler.state_dict()
    scheduler_state_str = json.dumps(scheduler_state, indent=4)

    save_name_txt = 'lr_scheduler_hyperparams.txt'
    save_path_name_txt = 'runs/{}/plots/{}'.format(ps_sim_name, save_name_txt)
    with open(save_path_name_txt, 'w') as file:
        file.write(scheduler_state_str)
