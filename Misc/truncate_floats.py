import torch
import sys

# This script will zeroize two least significant bytes on every value in a given f32 file
# This makes the file a lot more compressible at a cost of precision loss
#
# PyTorch will round when casting to bfloat16()
# This is used here to slightly reduce the precision loss vs plain zeroizing
#
assert len(sys.argv) == 2

with open(sys.argv[1], "rb") as f:
	f32_bytes = bytearray(f.read())

f32_tensor = torch.frombuffer(f32_bytes, dtype=torch.float)
bf16_tensor = f32_tensor.bfloat16()

truncated_f32_bytes = bytearray(len(f32_bytes))
truncated_f32_tensor = torch.frombuffer(truncated_f32_bytes, dtype=torch.float)
truncated_f32_tensor[:] = bf16_tensor

with open(sys.argv[1] + ".trunc", "wb") as f:
	f.write(truncated_f32_bytes)
