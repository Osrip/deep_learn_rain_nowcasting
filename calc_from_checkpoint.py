from helper.checkpoint_handling import load_from_checkpoint


if __name__ == '__main__':


    plot_settings = {
        'ps_runs_path': '/home/jan/jan/programming/first_CNN_on_Radolan/runs',
        'ps_run_name': 'Run_20230601-124534_test_profiler',
        'ps_checkpoint_name': 'model_epoch=0_val_loss=4.12.ckpt',
    }

    model = load_from_checkpoint(plot_settings['ps_runs_path'], plot_settings['ps_run_name'],
                                 plot_settings['ps_checkpoint_name'])



    #pass