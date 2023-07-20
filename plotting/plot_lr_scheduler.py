import matplotlib.pyplot as plt
import torch
import json
import warnings


class NullModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

def plot_lr_schedule(lr_scheduler, training_steps_per_epoch, epochs, ps_sim_name, init_learning_rate, save=True, **__):
    model = NullModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)



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
    if save:
        plt.savefig(save_path_name, dpi=200, bbox_inches='tight')
    # plt.yscale('log')
    plt.show()

    # Save hyperparams to txt
    if save:
        try:
            scheduler_state = lr_scheduler.state_dict()
            scheduler_state_str = json.dumps(scheduler_state, indent=4)

            save_name_txt = 'lr_scheduler_hyperparams.txt'
            save_path_name_txt = 'runs/{}/plots/{}'.format(ps_sim_name, save_name_txt)
            with open(save_path_name_txt, 'w') as file:
                file.write(scheduler_state_str)
        except Exception:
            warnings.warn('Failed to save hyperparameters of scheduler in plotting module')


if __name__ == '__main__':
    model = NullModule()
    optimizer = torch.optim.Adam(model.parameters())
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1-2*10e-6)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 3 * 10e-6)
    plot_lr_schedule(lr_scheduler, training_steps_per_epoch=1000, epochs=300, ps_sim_name=None, save=False,
                     init_learning_rate=0.001)
