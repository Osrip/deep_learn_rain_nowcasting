import matplotlib.pyplot as plt
import numpy as np
import gc


def plot_logged_metrics(train_df, val_df, train_mean_key, val_mean_key, train_std_key, val_std_key,
                        save_name, title, xlog, ylog,
                        s_dirs, **__):
    save_dir = s_dirs['save_dir']
    save_path_name = f'{save_dir}/plots/{save_name}.png'

    train_mean = train_df[train_mean_key].to_list()
    val_mean = val_df[val_mean_key].to_list()

    train_std = train_df[train_std_key].to_list()
    val_std = val_df[val_std_key].to_list()

    # TODO: Currently validatioon data is longer due to validatoon at epoch 0
    # if not len(train_mean) == len(val_mean):
    #     raise ValueError('Length of train and validation data is not the same in the logs')

    epochs_train = np.arange(0, len(train_mean))
    epochs_val = np.arange(0, len(val_mean))

    plt.figure()
    ax = plt.subplot(111)
    plt.title(title)
    ax.plot(epochs_train, train_mean, label='train', color='blue')
    ax.plot(epochs_val, val_mean, label='validation', color='red')

    # Add shaded area for standard deviations
    # ax.fill_between(epochs, np.array(train_mean) - np.array(train_std), np.array(train_mean) + np.array(train_std), color='blue', alpha=0.2)
    # ax.fill_between(epochs, np.array(val_mean) - np.array(val_std), np.array(val_mean) + np.array(val_std), color='red', alpha=0.2)

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


def plot_mean_predictions(train_df, val_df,
                          train_mean_pred_key, val_mean_pred_key, train_std_pred_key, val_std_pred_key,
                          train_mean_target_key, val_mean_target_key, train_std_target_key, val_std_target_key,
                          save_name, title,
                          xlog, ylog, s_dirs, **__):

    save_dir = s_dirs['save_dir']
    save_path_name = f'{save_dir}/plots/{save_name}.png'

    train_pred_means = train_df[train_mean_pred_key].to_list()
    val_pred_means = val_df[val_mean_pred_key].to_list()
    train_pred_stds = train_df[train_std_pred_key].to_list()
    val_pred_stds = val_df[val_std_pred_key].to_list()

    train_target_means = train_df[train_mean_target_key].to_list()
    val_target_means = val_df[val_mean_target_key].to_list()
    train_target_stds = train_df[train_std_target_key].to_list()
    val_target_stds = val_df[val_std_target_key].to_list()

    if not (len(train_pred_means) == len(train_target_means) == len(train_pred_stds) == len(train_target_stds)):
        raise ValueError("Length of train data is not consistent in the logs")

    if not (len(val_pred_means) == len(val_target_means) == len(val_pred_stds) == len(val_target_stds)):
        raise ValueError("Length of validation data is not consistent in the logs")

    epochs_train = np.arange(0, len(train_pred_means))
    epochs_val = np.arange(0, len(val_pred_means))

    plt.figure()
    ax = plt.subplot(111)
    plt.title(title)
    ax.plot(epochs_train, train_pred_means, label='prediction training', color='blue')
    ax.plot(epochs_train, train_target_means, label='target training', color='blue', linestyle='dashed')
    ax.plot(epochs_val, val_pred_means, label='prediction validation', color='red')
    ax.plot(epochs_val, val_target_means, label='target validation', color='red', linestyle='dashed')

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







