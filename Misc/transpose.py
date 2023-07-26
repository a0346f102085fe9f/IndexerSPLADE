import torch
import json

# This script will perform some reversible filtering on the 30000x30522 float matrix produced sparse2dense.py
# 1. The rows are sorted such that the rows that have the most nonzero elements come first
# 2. The matrix is transposed, which results in more consecutive zeros and makes gzip a little happier
#

with open("datatape.f32", "rb") as f:
	b_src = bytearray(f.read())

assert len(b_src) % (4*30522) == 0

n = len(b_src) // (4*30522)
b_dst = bytearray(n*30522*4)

x = torch.frombuffer(b_src, dtype=torch.float).view([n, 30522])
t = torch.frombuffer(b_dst, dtype=torch.float).view([30522, n])
nz_counts = torch.tensor([row.count_nonzero() for row in x])
s_nz_count, indices = nz_counts.sort(descending=True)

x[:] = x[indices]
t[:] = x.T

#with open("s_datatape.f32", "wb") as f:
#	f.write(b_src)

with open("st_datatape.f32", "wb") as f:
	f.write(b_dst)

with open("idx.json", "r") as f:
	idx = json.load(f)

titles = list(idx.keys())
remap_idx = {}

for i, j in enumerate(indices.tolist()):
	title = titles[j]
	remap_idx[title] = idx[title]

with open("idx_remap.json", "w") as f:
	json.dump(remap_idx, f)
