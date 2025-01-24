import xarray as xr
import zarr
import argparse

def rechunk_time_dimension(input_zarr, output_zarr):

    # Open the input Zarr dataset
    ds = xr.open_zarr(input_zarr, chunks={'step': 1, 'time': 200, 'y': 1200, 'x': 1100})

    # Re-chunk the dataset: set time chunk size to 1
    #ds_rechunked = ds.chunk(chunks={'step': 1, 'time': 1, 'y': 1200, 'x': 1100})

    for var in ds.variables:
        if 'chunks' in ds[var].encoding:
            print(f"Deleting chunk encoding for {var}", flush=True)
            del ds[var].encoding['chunks']
    ds = ds.chunk({'step': 1,'time': 1,'y': 1200, 'x': 1100})

    # Save the rechunked dataset to a new Zarr store
    ds.to_zarr(output_zarr, mode="w", consolidated=True)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Rechunk the time dimension of a Zarr dataset.")
    parser.add_argument("input_zarr", type=str, help="Path to the input Zarr dataset.")
    parser.add_argument("output_zarr", type=str, help="Path to save the rechunked Zarr dataset.")

    # Parse arguments
    args = parser.parse_args()

    # Perform the rechunking
    rechunk_time_dimension(args.input_zarr, args.output_zarr)

    print("Rechunking completed. The rechunked dataset is saved to:", args.output_zarr)