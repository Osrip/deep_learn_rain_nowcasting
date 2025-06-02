import zarr

# Access the Zarr store directly
zarr_store = zarr.open('optical_flow_2019_2020.zarr')

# Print the structure of the Zarr store
print("Zarr store top-level contents:")
print(list(zarr_store))

# Explore the first level
for name in zarr_store:
    item = zarr_store[name]
    print(f"\nExamining: {name}")
    print(f"Type: {type(item)}")

    # If it's an array, print chunk info
    if isinstance(item, zarr.core.Array):
        print(f"  Chunks: {item.chunks}")
        print(f"  Shape: {item.shape}")
        print(f"  Dtype: {item.dtype}")

    # If it's a group, list its contents
    elif isinstance(item, zarr.hierarchy.Group):
        print(f"  Group contents: {list(item)}")

        # Check if RV_recalc is in this group
        if 'RV_recalc' in item:
            arr = item['RV_recalc']
            print(f"  Found RV_recalc array!")
            print(f"  Chunks: {arr.chunks}")
            print(f"  Shape: {arr.shape}")
            print(
                f"  Chunk size: {arr.chunks[0] * arr.chunks[1] * arr.chunks[2] * arr.chunks[3] * arr.dtype.itemsize / (1024 ** 2):.2f} MB per chunk")