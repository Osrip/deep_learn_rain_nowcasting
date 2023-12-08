from hurry.filesize import size
import psutil
import torch

def print_gpu_memory():
    # Check if CUDA is available and get the current device
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

        # Get total memory
        total_memory = torch.cuda.get_device_properties(device).total_memory

        # Get current memory usage
        current_memory = torch.cuda.memory_allocated(device)

        print(f"\nCUDA device {device} Total memory : {size(total_memory)} Used memory: {size(current_memory)}")


def print_ram_usage():
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used:', size(psutil.virtual_memory()[3]))
