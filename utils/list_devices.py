"""List all audio devices and host APIs"""
import sounddevice as sd

print("\n=== All audio devices ===")
print("-" * 100)
for i, dev in enumerate(sd.query_devices()):
    name = dev["name"]
    host_api = sd.query_hostapis()[dev["hostapi"]]["name"]
    ins = dev["max_input_channels"]
    outs = dev["max_output_channels"]
    sr = dev["default_samplerate"]
    print(f"  [{i}] {name}")
    print(f"      API: {host_api}, input: {ins}ch, output: {outs}ch, sample rate: {sr}")

print("\n=== Host APIs ===")
for i, api in enumerate(sd.query_hostapis()):
    print(f"  [{i}] {api['name']} (devices: {len(api['devices'])})")

print()
