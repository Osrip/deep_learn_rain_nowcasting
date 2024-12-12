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
        # current_memory = torch.cuda.memory_allocated(device)
        max_memory = torch.cuda.max_memory_allocated(device)

        print(f"\nCUDA device {device} Total memory : {size(total_memory)} Max memory used: {size(max_memory)}")


# def print_ram_usage():
#     # Getting % usage of virtual_memory ( 3rd field)
#     print('RAM memory % used:', psutil.virtual_memory()[2])
#     # Getting usage of virtual_memory in GB ( 4th field)
#     print('RAM Used:', size(psutil.virtual_memory()[3]))

def print_ram_usage():
    # Getting percentage usage of virtual memory
    print('RAM memory % used:', psutil.virtual_memory().percent)
    # Getting usage of virtual memory in GB
    used_ram = psutil.virtual_memory().used / (1024 ** 3)
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"RAM Used: {used_ram:.2f} GB out of {total_ram:.2f} GB")

def format_ram_usage():
    """Formats the current RAM usage in a readable string."""
    used_ram = psutil.virtual_memory().used / (1024 ** 3)
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    return f"RAM before starting: {used_ram:.2f} GB out of {total_ram:.2f} GB ({psutil.virtual_memory().percent}%)"

def format_duration(duration):
    """Formats a duration in seconds to a human-readable string."""
    if duration < 60:
        return f"{duration:.2f} seconds"
    elif duration < 3600:
        return f"{duration / 60:.2f} minutes"
    else:
        return f"{duration / 3600:.2f} hours"


