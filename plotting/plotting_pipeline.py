
from plotting.plot_lr_scheduler import plot_lr_schedule

from helper.plotting_helper import load_data_from_logs
from plotting.plot_from_log import plot_logged_metrics, plot_mean_predictions


def plot_logs_pipeline(training_steps_per_epoch, model_l, settings,
                       plot_lr_schedule_boo=True, **__):
    '''
    Pipeline for automatic plotting of several figures
    For plot_lr_schedule=True, model_l is required, otherwise None can be passed for model_l
    '''

    ###### Plot logged metrics ######

    key_list_train_mean = ['train_mean_loss', 'train_mean_rmse']
    key_list_val_mean = ['val_mean_loss', 'val_mean_rmse']
    key_list_train_std = ['train_std_loss', 'train_std_rmse']
    key_list_val_std = ['val_std_loss', 'val_std_rmse']
    save_name_list = ['loss', 'mse']
    title_list = ['Loss', 'RMSE (mm)']

    train_df, val_df, base_train_df, base_val_df = load_data_from_logs(**settings)

    for train_mean_key, val_mean_key, train_std_key, val_std_key, save_name, title in\
            zip(key_list_train_mean, key_list_val_mean, key_list_train_std, key_list_val_std, save_name_list, title_list):
        plot_logged_metrics(
            train_df, val_df,
            train_mean_key, val_mean_key, train_std_key, val_std_key,
            save_name, title,
            xlog=False, ylog=True, **settings)

    ###### Plot mean predictions ######

    plot_mean_predictions(train_df, val_df,
                          train_mean_pred_key='train_mean_mean_pred',
                          val_mean_pred_key='val_mean_mean_pred',
                          train_std_pred_key='train_std_mean_pred',
                          val_std_pred_key='val_std_mean_pred',
                          train_mean_target_key='train_mean_mean_target',
                          val_mean_target_key='val_mean_mean_target',
                          train_std_target_key='train_std_mean_target',
                          val_std_target_key='val_std_mean_target',
                          save_name='mean_predictions_targets', title='Mean precipitation (mm)',
                          xlog=False, ylog=True, **settings)

    ###### PLot lr scheduler ######

    if settings['s_lr_schedule'] and plot_lr_schedule_boo:

        plot_lr_schedule(model_l.lr_scheduler, training_steps_per_epoch, settings['s_max_epochs'],
                         save_name='lr_scheduler', y_label='Learning Rate', title='LR scheduler',
                         ylog=True, **settings)






