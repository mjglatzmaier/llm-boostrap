"""GPU diagnostics: PyTorch device properties + live pynvml stats."""

import sys

try:
    import torch
except ImportError:
    print("PyTorch is not installed in this environment.")
    sys.exit(1)

try:
    import pynvml
    _PYNVML_OK = True
except ImportError:
    _PYNVML_OK = False


def fmt_gb(n: int) -> str:
    return f"{n / 1024**3:.1f} GB"


print(f"Python     : {sys.version.split()[0]}")
print(f"Executable : {sys.executable}")
print(f"PyTorch    : {torch.__version__}")
print(f"CUDA avail : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nNo CUDA-capable GPU detected — inference will run on CPU.")
    sys.exit(0)

n_gpus = torch.cuda.device_count()
print(f"GPU count  : {n_gpus}")

for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"\n  [{i}] {props.name}")
    print(f"      Compute capability : {props.major}.{props.minor}")
    print(f"      Total VRAM         : {fmt_gb(props.total_memory)}")
    print(f"      Multi-processors   : {props.multi_processor_count}")

if _PYNVML_OK:
    try:
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        print(f"\nNVIDIA driver : {driver}")

        for i in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000

            print(f"\n  [{i}] Live stats:")
            print(f"      GPU utilization    : {util.gpu} %")
            print(f"      Memory utilization : {util.memory} %")
            print(f"      Temperature        : {temp} °C")
            print(f"      Power draw         : {power_w:.0f} W / {power_limit_w:.0f} W")
            print(f"      VRAM used          : {fmt_gb(mem_info.used)} / {fmt_gb(mem_info.total)}")
    except Exception as exc:
        print(f"\npynvml error: {exc}")
else:
    print("\n(Install pynvml for live GPU utilization, temperature, and power stats.)")
