import torch
import json

# This script will convert the index as produced by run_inference.py into a contiguous Nx30522 array of floats
#
# Takes:
# - idx.json
# - datatape_k.bin
# - datatape_v.bin
#
# Produces:
# - datatape.f32
#
# For N = 30000, the resulting file size will be:
# 30000x30522x4 bytes = 3.41 G
#
with open("idx.json") as f:
	idx = json.load(f)

keys = open("datatape_k.bin", "rb")
vals = open("datatape_v.bin", "rb")
out = open("datatape.f32", "wb")

buffer = bytes(4*30522)
dense = torch.frombuffer(buffer, dtype=torch.float)

for title, spec in idx.items():
	dim = spec["dimensions"]

	b_k = keys.read(2*dim)
	b_v = vals.read(4*dim)

	k = torch.frombuffer(b_k, dtype=torch.int16).int()
	v = torch.frombuffer(b_v, dtype=torch.float)

	dense[:] = 0.0
	dense[k] = abs(v) # Wrap v in abs() to remove strayness information

	out.write(buffer)
	print(dim)

keys.close()
vals.close()
out.close()
