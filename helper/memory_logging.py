from hurry.filesize import size
import psutil

def print_gpu_memory():
    import nvidia_smi
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("GPU Memory: Total: {}, Free: {}, Used:{}".format(size(info.total), size(info.free), size(info.used)))


def print_ram_usage():
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used:', size(psutil.virtual_memory()[3]))
