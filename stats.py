import psutil
import subprocess
from statistics import mean


def get_gpu_stats():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        gpu_stats = output.decode().strip().split("\n")
        gpu_utilization = mean([int(stats.split(",")[0]) for stats in gpu_stats])
        gpu_memory_used = mean([int(stats.split(",")[1]) for stats in gpu_stats])
        gpu_memory_total = mean([int(stats.split(",")[2]) for stats in gpu_stats])
        return (
            gpu_utilization,
            round(gpu_memory_used / 1e3, 2),
            round(gpu_memory_total / 1e3, 2),
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0, 0, 0


def get_system_stats():
    system_stats = {}

    # Get CPU information
    cpu_count = psutil.cpu_count(logical=True)  # Physical CPU cores
    cpu_percent = psutil.cpu_percent(
        interval=1, percpu=False
    )  # CPU utilization percentage per core

    # Get memory information
    memory = psutil.virtual_memory()
    total_memory = memory.total  # Total memory in bytes
    used_memory = memory.used  # Used memory in bytes
    memory_percent = memory.percent  # Memory utilization percentage

    # Add CPU information to the dictionary
    system_stats["cpu_cores"] = cpu_count
    system_stats["cpu_utilization"] = cpu_percent

    # Add memory information to the dictionary
    system_stats["memory"] = round(total_memory / (1024**3), 2)  # Convert to GB
    system_stats["memory_utilization"] = memory_percent

    # Get disk utilization
    disk_usage = psutil.disk_usage("/")
    total_disk_space = disk_usage.total  # Total disk space in bytes
    used_disk_space = disk_usage.used  # Used disk space in bytes
    disk_percent = disk_usage.percent  # Disk utilization percentage
    system_stats["disk"] = round(total_disk_space / (1024**3), 2)  # Convert to GB
    system_stats["disk_utilization"] = disk_percent

    # Get GPU utilization
    gpu_utilization, gpu_memory_used, gpu_memory_total = get_gpu_stats()
    system_stats["gpu_utilization"] = gpu_utilization
    system_stats["gpu_memory_used"] = gpu_memory_used
    system_stats["gpu_memory"] = gpu_memory_total

    return system_stats
