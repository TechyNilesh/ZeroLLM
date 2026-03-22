"""Hardware detection — see what ZeroLLM detects about your machine."""

from zerollm.hardware import detect, compute_n_gpu_layers, compute_threads

# Detect your hardware
hw = detect()

print(f"Platform:   {hw.platform} ({hw.arch})")
print(f"CPU:        {hw.cpu}")
print(f"RAM:        {hw.ram_gb}GB")
print(f"GPU:        {hw.gpu_type or 'None'}")
print(f"GPU Name:   {hw.gpu_name or 'N/A'}")
print(f"GPU VRAM:   {hw.gpu_vram_gb or 'N/A'}GB")
print(f"Threads:    {hw.n_threads}")
print(f"Has GPU:    {hw.has_gpu}")
print(f"Summary:    {hw.summary()}")

# See how power setting affects resource allocation
print("\nPower settings (assuming 40-layer model):")
for power in [0.0, 0.25, 0.5, 0.75, 1.0]:
    layers = compute_n_gpu_layers(40, power)
    threads = compute_threads(power, hw)
    print(f"  power={power:.2f} → GPU layers: {layers:>3}, CPU threads: {threads}")
