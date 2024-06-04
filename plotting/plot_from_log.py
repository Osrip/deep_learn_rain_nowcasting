import matplotlib.pyplot as plt
import numpy as np
import gc



def plot_qualities_main(plot_settings, s_calc_baseline, **__):
    '''
    This functions plots logged metrics
    pass **settings and **plot_settings
    '''



    pass


def plot_logged_metrics(train_key, val_key, save_name, title, train_df, val_df, xlog, ylog, s_dirs, **__):
    save_dir = s_dirs['save_dir']
    save_path_name = f'{save_dir}/plots/{save_name}.png'

    train_data = train_df[train_key].to_list()
    val_data = val_df[val_key].to_list()

    if not len(train_data) == len(val_data):
        raise ValueError('Length of train and validation data is not the same in the logs')

    epochs = np.arange(0, len(train_data))

    plt.figure()
    ax = plt.subplot(111)
    plt.title(title)
    ax.plot(epochs, train_data, label='train', color='blue')
    ax.plot(epochs, val_data, label='validation', color='red')

    if ylog:
        ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(save_path_name, dpi=200, bbox_inches='tight')
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()








