import json
import platform
import shutil
import subprocess
import sys


def _nvidia_smi_query():
    fields = "name,memory.total,memory.used,memory.free,driver_version"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    line = out.stdout.strip().splitlines()[0]
    name, mem_total, mem_used, mem_free, driver = [p.strip() for p in line.split(",")]
    return {
        "name": name,
        "vram_total_mb": int(mem_total),
        "vram_used_mb": int(mem_used),
        "vram_free_mb": int(mem_free),
        "driver_version": driver,
    }


def _torch_cuda_check():
    try:
        import torch
    except ImportError:
        return {"torch_installed": False}
    info = {"torch_installed": True, "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info["cuda_device_name"] = props.name
        info["cuda_total_vram_mb"] = props.total_memory // (1024 * 1024)
    return info


def _ram_info():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total_mb": vm.total // (1024 * 1024), "available_mb": vm.available // (1024 * 1024)}
    except ImportError:
        # psutil not yet installed at first preflight run; fall back to /proc on Linux
        # or a conservative unknown on Windows without the dependency.
        if platform.system() == "Linux":
            meminfo = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":")
                    meminfo[k.strip()] = int(v.strip().split()[0])
            return {"total_mb": meminfo["MemTotal"] // 1024, "available_mb": meminfo["MemAvailable"] // 1024}
        return {"total_mb": None, "available_mb": None, "note": "psutil not installed"}


def _disk_free_gb(path="."):
    total, used, free = shutil.disk_usage(path)
    return round(free / (1024 ** 3), 2)


def main():
    report = {
        "os": platform.platform(),
        "python_version": sys.version,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "ram": _ram_info(),
        "disk_free_gb": _disk_free_gb("."),
        "gpu_nvidia_smi": _nvidia_smi_query(),
        "torch": _torch_cuda_check(),
    }
    min_disk_gb = 25
    report["disk_ok"] = report["disk_free_gb"] >= min_disk_gb
    print(json.dumps(report, indent=2))
    if not report["disk_ok"]:
        sys.stderr.write(f"insufficient disk: {report['disk_free_gb']} GB free, need {min_disk_gb} GB\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
