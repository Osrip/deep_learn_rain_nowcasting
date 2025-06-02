import xarray as xr
import numpy as np
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
import os
import imageio

def save_lead_time_steps_as_gif(zarr_path, output_gif, time_idx=0, duration=0.5):
    # Load the Zarr dataset
    radolan_pred_ds = xr.open_zarr(zarr_path)
    
    # Extract the specified time index
    forecast_slice = radolan_pred_ds.isel(time=time_idx)
    
    # List to store the file paths of generated frames
    frame_images = []
    
    # Iterate over each lead time step
    num_lead_times = forecast_slice.dims['lead_time']
    for lead_idx in range(num_lead_times):
        print(f"Processing lead time step {lead_idx} for time index {time_idx}")
        
        # Extract the precipitation field for the current lead time
        precip_field = forecast_slice['extrapolation'].isel(lead_time=lead_idx).values
        
        # Create a plot for the precipitation field
        plt.figure(figsize=(10, 8))
        title = f"Time Index {time_idx}, Lead Time Step {lead_idx}"
        plot_precip_field(
            precip_field,
            geodata=None,  # Add geodata if available
            units="mm/h",
            title=title,
            colorbar=True,
            axis="off"
        )
        
        # Save the plot to a temporary file
        temp_image_path = f"frame_{lead_idx}.png"
        plt.savefig(temp_image_path, bbox_inches="tight")
        plt.close()
        frame_images.append(temp_image_path)
    
    # Create the GIF
    print(f"Creating GIF: {output_gif}")
    with imageio.get_writer(output_gif, mode="I", duration=duration) as writer:
        for frame_path in frame_images:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Remove the temporary images
    for frame_path in frame_images:
        os.remove(frame_path)
    print(f"GIF saved as {output_gif}")

if __name__ == "__main__":
    # Path to the Zarr dataset
    zarr_path = './extrapolation_2019_2020.zarr'
    # Output GIF file path
    output_gif = './extrapolation.gif'
    # Time index to process
    time_idx = 2412
    
    # Save the lead time steps as a GIF
    save_lead_time_steps_as_gif(zarr_path, output_gif, time_idx)
