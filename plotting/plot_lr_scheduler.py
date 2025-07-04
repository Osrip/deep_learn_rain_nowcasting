import matplotlib.pyplot as plt
import torch
import json
import warnings


class NullModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)


def plot_lr_schedule(lr_scheduler, training_steps_per_epoch, epochs, s_dirs, save=True,
                     save_name='lr_scheduler', y_label='Learning Rate', title='LR scheduler', ylog=False, **__):
    # TODO init learing rate seems to do nothing!, Init learning rate is adapted fropm the optimizer, that is fed in!

    ps_sim_name = s_dirs['save_dir']

    model = NullModule()
    optimizer = torch.optim.Adam(model.parameters())

    lrs = []
    for _ in range(epochs * training_steps_per_epoch):
        optimizer.step()
        lrs.append(lr_scheduler.get_lr())
        lr_scheduler.step()
    plt.clf()
    plt.plot(lrs)
    save_path_name = '{}/plots/{}'.format(ps_sim_name, save_name)
    plt.ylabel(y_label)
    plt.xlabel('Step (total corresponds to steps in training)')
    plt.title('{}: {}'.format(title, lr_scheduler.__class__.__name__))
    if ylog:
        plt.yscale('log')
    if save:
        plt.savefig(save_path_name, dpi=200, bbox_inches='tight')

    plt.show()

    # Save hyperparams to txt
    if save:
        try:
            scheduler_state = lr_scheduler.state_dict()
            scheduler_state_str = json.dumps(scheduler_state, indent=4)

            save_name_txt = 'lr_scheduler_hyperparams.txt'
            save_path_name_txt = '{}/plots/{}'.format(ps_sim_name, save_name_txt)
            with open(save_path_name_txt, 'w') as file:
                file.write(scheduler_state_str)
        except Exception:
            warnings.warn('Failed to save hyperparameters of scheduler in plotting module')



def plot_sigma_schedule(sigma_schedule_mapping, ps_sim_name, save_name='sigma_scheduler', ylog=False, save=True, **__):
    save_path_name = '{}/plots/{}'.format(ps_sim_name, save_name)

    plt.clf()
    plt.plot(sigma_schedule_mapping)
    plt.ylabel('Sigma')
    plt.xlabel('Step (total corresponds to steps in training)')
    plt.title('Sigma schedule')
    if ylog:
        plt.yscale('log')
    if save:
        plt.savefig(save_path_name, dpi=200, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    model = NullModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    training_steps_per_epoch = 6000
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 1 * (1/epochs * (6000/training_steps_per_epoch)) * 10e-4)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 1 * (1/epochs) * 10e-4)
    # Gamma = 1 - x * (1 / epochs) keeps exponential equal independently of value for epochs

    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 3 * 10e-6)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1-2*10e-6)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 3 * 10e-6)
    plot_lr_schedule(lr_scheduler, training_steps_per_epoch=training_steps_per_epoch, epochs=epochs, ps_sim_name=None, save=False,
                     ylog=True)
