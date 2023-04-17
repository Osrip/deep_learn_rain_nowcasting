import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_image(image, save_path_name, vmin, vmax, title=''):
    plt.figure()
    plt.title(title)
    image = np.array(image.cpu())
    im = plt.imshow(image, vmin=vmin, vmax=vmax, norm='linear')
    plt.colorbar(im)
    plt.savefig(save_path_name, dpi=200)
    plt.show()


def plot_target_vs_pred(target_img, pred_img, save_path_name, vmin, vmax, max_row_num=3, title=''):
    convert_tensor_to_np = lambda x: x.cpu().detach().numpy()
    target_img = convert_tensor_to_np(target_img)
    pred_img = convert_tensor_to_np(pred_img)
    num_rows = np.min((target_img.shape[0], max_row_num))
    num_cols = 2
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5*num_rows, 5*num_cols))

    for row in range(num_rows):
        for col in range(num_cols):
            curr_ax = axs[row, col]
            curr_ax.imshow(target_img[row, :, :] if col == 0 else pred_img[row, :, :], vmin=vmin, vmax=vmax, norm='linear')
            if row == 0:
                if col == 0:
                    curr_ax.set_title('Targets')
                elif col == 1:
                    curr_ax.set_title('Predictions')
    # plt.colorbar(fig)
    plt.savefig(save_path_name, dpi=600)
    plt.show()




# def plot_image_log(image, save_path_name, vmin, vmax, title=''):
#     fig = plt.figure(figsize=(10, 10))
#     plt.title(title)
#     image = np.array(image.cpu())
#     im = plt.imshow(image, vmin=vmin, vmax=vmax)
#     plt.colorbar(im)
#     plt.savefig(save_path_name, dpi=200)
#     plt.show()
