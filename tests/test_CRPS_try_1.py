import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def element_wise_crps(bin_probs, observation, bin_edges):
    """
    Calculate CRPS between an empirical distribution and a point observation.
    Parameters:
    - bin_edges : array-like, bin edges including leftmost and rightmost edge
    - bin_probs : array-like, probabilities of each bin (len(bin_probs) == len(bin_edges) - 1)
    - observation : float, observed value
    Returns:
    - CRPS value : float
    """
    cdf = np.cumsum(bin_probs)
    crps = 0
    for i in range(len(bin_edges) - 1):
        left_edge = bin_edges[i]
        right_edge = bin_edges[i + 1]
        if observation > right_edge:
            crps += cdf[i] ** 2 * (right_edge - left_edge)
        else:
            crps += (cdf[i] - 1) ** 2 * (right_edge - left_edge)
    return crps

# Function to calculate CRPS for the entire dataset
def iterate_crps(pred_np, target_inv_normed, bin_edges):
    shape_pred = pred_np.shape
    shape_target = target_inv_normed.shape

    if not (shape_target[0] == shape_pred[0] and shape_target[1] == shape_pred[2] and shape_target[2] == shape_pred[3]):
        raise ValueError('Dimensionality mismatch between prediction and target')

    crps_out = np.zeros(shape_target)
    for b in range(shape_target[0]):
        for h in range(shape_target[1]):
            for w in range(shape_target[2]):
                crps = element_wise_crps(pred_np[b, :, h, w], target_inv_normed[b, h, w], bin_edges)
                crps_out[b, h, w] = crps
    return crps_out


# Function to generate initial random test data
def generate_initial_data(batch_size, height=32, width=32):
    return np.random.rand(batch_size, height, width)

# Function to apply Gaussian blurring and add offsets
def modify_data(data, std_dev, offset):
    blurred_data = gaussian_filter(data, sigma=std_dev)
    offset_data = blurred_data + offset
    return offset_data

# Function to create prediction probability bins
def create_prediction_bins(data, std_dev, bin_edges):
    bins = np.zeros((data.shape[0], len(bin_edges)-1, data.shape[1], data.shape[2]))
    for b in range(data.shape[0]):
        for h in range(data.shape[1]):
            for w in range(data.shape[2]):
                pixel_value = data[b, h, w]
                # Binning around the pixel value using std_dev
                bin_probs = np.exp(-0.5 * ((bin_edges - pixel_value) / std_dev) ** 2)
                bin_probs /= bin_probs.sum()  # Normalizing to get probabilities
                bins[b, :, h, w] = bin_probs[:-1]  # Excluding the rightmost edge
    return bins

# Binning definition
linspace_binning_inv_norm = np.array([
    -0.02084237,  0.01762839,  0.05761065,  0.0991638,   0.14234956,  0.18723208,
    0.23387801,  0.28235665,  0.33273999,  0.38510288,  0.43952308,  0.49608144,
    0.55486196,  0.61595194,  0.67944213,  0.74542682,  0.81400403,  0.88527561,
    0.95934743,  1.03632951,  1.11633618,  1.19948629,  1.28590334,  1.37571568,
    1.46905672,  1.56606509,  1.66688489,  1.77166587,  1.88056365,  1.99373999,
    2.11136298,  2.23360735,  2.36065465,  2.49269359,  2.6299203,   2.7725386,
    2.92076032,  3.07480563,  3.23490331,  3.40129118,  3.57421637,  3.75393573,
    3.94071621,  4.13483522,  4.3365811,   4.5462535,   4.76416386,  4.99063584,
    5.22600583,  5.47062342,  5.72485195,  5.98906903,  6.26366711,  6.54905405,
    6.84565374,  7.15390673,  7.47427087,  7.80722201,  8.15325468,  8.51288286,
    8.88664069,  9.27508335,  9.67878778, 10.09835361
])
linspace_binning_max_inv_norm = 10.534404044730364

# Extend binning edges to include the maximum value
bin_edges = np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm)

# Adjustable variables for test data
batch_size = 10  # Batch size
std_devs = [1, 2, 3, 4, 5]  # Different standard deviations for Gaussian blurring
offsets = np.linspace(0, 10, 30)  # 30 different offsets

# Generate initial data
target_inv_normed = generate_initial_data(batch_size)

# Create figure for the plot
plt.figure(figsize=(12, 8))

# Iterate over different standard deviations
for std_dev in std_devs:
    mean_crps = []
    std_crps = []

    # Iterate over different offsets
    for offset in offsets:
        # Modify data
        modified_data = modify_data(target_inv_normed, std_dev, offset)

        # Create prediction bins
        pred_np = create_prediction_bins(modified_data, std_dev, bin_edges)

        # Calculate CRPS
        crps_values = iterate_crps(pred_np, modified_data, bin_edges)
        mean_crps.append(np.mean(crps_values))
        std_crps.append(np.std(crps_values))

    # Plotting
    plt.plot(offsets, mean_crps, label=f'Std Dev {std_dev}')
    plt.fill_between(offsets, np.array(mean_crps) - np.array(std_crps), np.array(mean_crps) + np.array(std_crps), alpha=0.3)

# Finalizing plot
plt.xlabel('Offset')
plt.ylabel('CRPS')
plt.title('CRPS vs Offset for Different Standard Deviations')
plt.legend()
plt.show()


# Continue from where the code left off

'''
Write test:

I want to test my CRPS function. Can you write a function that creates test data? I would to create a random_test_data_set. target_inv_normed contain the direct values of the initial random data set.  target_inv_normed contains one sample and I want the samples to be an adjustable variable
target np should have dimensions b x h x w so batch (adjustable varibale as well) x 32 x 32

For pred_np samples contained in target_inv_normed should have an added gaussian blurring. The magnitude of the standard deviation as well as the magitude should be varied for each std_modified_data_set (containing a several std modified test_data_set s). Additionally there should be an offset_modified_data_set (containining several offset modified std_modified_data_set s)  created with an added offset (just adding a number to each value to create offset). This way the test data has two variations from the inititial random test data: std gaussian blurring adding and offset adding.  

Now pred_np has a probability binning for each pixel. It has dimensions b x c x h x w where c is the binning dimension, thus b x 64 x 32 x 32 thus 64 bins. Use the std that was used to modify the current set within the std_modified_data_set to create the binning around the mean of the current pixel.

Here are the functions to be tested:

def iterate_crps(pred_np, target_inv_normed, linspace_binning_inv_norm, linspace_binning_max_inv_norm):

    # pred_np: binned prediction b x c x h x w
    # target_inv_normed: target in inv normed space b x h x w
    # linspace_binning_inv_norm: left bins edges in inv normed space

    calculate_crps_lambda = lambda x, y: element_wise_crps(x, y,
                                                           np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm),)
    shape_pred = np.shape(pred_np)
    shape_target = np.shape(target_inv_normed)
    if not (shape_target[0] == shape_pred[0] and shape_target[1] == shape_pred[2] and shape_target[2] == shape_pred[3]):
        raise ValueError('Dimensionality mismatch between prediction and target (leaving away channel dimension of prediction')

    crps_out = np.zeros(shape=shape_target)

    for b in range(shape_target[0]):
        for h in range(shape_target[1]):
            for w in range(shape_target[2]):
                crps = calculate_crps_lambda(pred_np[b, :, h, w], target_inv_normed[b, h, w])
                crps_out[b, h, w] = crps
    return crps_out


def element_wise_crps(bin_probs, observation, bin_edges):
    """
    VIDEO ZUR IMPLEMENTATION IN ICLOUD NOTES UNTER NOTIZ "CRPS"
    Calculate CRPS between an empirical distribution and a point observation.
    Parameters:
    - bin_edges : array-like, bin edges
    !!including leftmost and rightmost edge!!!
    - bin_probs : array-like, probabilities of each bin --> len(bin_probs) == len(bin_edges - 1) as last right bin not included!
    - observation : float, observed value
    Returns:
    - CRPS value : float
    """
    # TODO Speed this up with jit!
    cdf = np.cumsum(bin_probs)
    crps = 0
    # Iterating through each bin and looking whether observation is outside
    for i in range(len(bin_edges)-1):
        left_edge = bin_edges[i]
        right_edge = bin_edges[i+1]
        if observation > right_edge:
            crps += cdf[i] ** 2 * (right_edge - left_edge)
            # Eveything smaller than observation is added to represent integral

        # elif observation < right_edge:
        else:
            crps += (cdf[i] - 1) ** 2 * (right_edge - left_edge)
            # For the bin that the observation is in and all larger bins Observation - 1 is added

    return crps

linspace_binning_inv_norm =
[-0.02084237  0.01762839  0.05761065  0.0991638   0.14234956  0.18723208,  0.23387801  0.28235665  0.33273999  0.38510288  0.43952308  0.49608144,  0.55486196  0.61595194  0.67944213  0.74542682  0.81400403  0.88527561,  0.95934743  1.03632951  1.11633618  1.19948629  1.28590334  1.37571568,  1.46905672  1.56606509  1.66688489  1.77166587  1.88056365  1.99373999,  2.11136298  2.23360735  2.36065465  2.49269359  2.6299203   2.7725386,  2.92076032  3.07480563  3.23490331  3.40129118  3.57421637  3.75393573,  3.94071621  4.13483522  4.3365811   4.5462535   4.76416386  4.99063584,  5.22600583  5.47062342  5.72485195  5.98906903  6.26366711  6.54905405,  6.84565374  7.15390673  7.47427087  7.80722201  8.15325468  8.51288286,  8.88664069  9.27508335  9.67878778 10.09835361]

linspace_binning_max_inv_norm =
10.534404044730364

I would like to end up with a line plot where offset is on the x axis and the CRPS is on the y axis (mean represents line and std is a shaded area around the line in the same color as the line)
Different stds correspond to different colors of the lines.
Let's start with 30 different offsets and 5 different means.

'''

