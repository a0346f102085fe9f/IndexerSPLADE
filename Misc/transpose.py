from mmap import mmap
import torch
import json

# This script will perform some reversible filtering on the 30000x30522 float matrix produced sparse2dense.py
# 1. The rows are sorted such that the rows that have the most nonzero elements come first
# 2. The matrix is transposed, which results in more consecutive zeros and makes gzip a little happier
#

print("Startup")

src_file = open("datatape.f32", "r+b")
src_mmap = mmap(src_file.fileno(), 0)

assert len(src_mmap) % (4*30522) == 0
n = len(src_mmap) // (4*30522)

src = torch.frombuffer(src_mmap, dtype=torch.float).view([n, 30522])
nz_counts = torch.tensor([row.count_nonzero() for row in src])
s_nz_count, indices = nz_counts.sort(descending=True)

print("Sorted")

with open("s_datatape.f32", "w+b") as f:
	f.truncate(n*4*30522)

	for i in indices.tolist():
		start = i*4*30522
		stop = start + 4*30522
		f.write(src_mmap[start:stop])

print("Saved")

del src
src_mmap.close()
src_file.close()

# Transpose by pieces
sorted_file = open("s_datatape.f32", "r+b")
sorted_mmap = mmap(sorted_file.fileno(), 0)

sorted = torch.frombuffer(sorted_mmap, dtype=torch.float).view([n, 30522])

t_mem = bytearray(4*n)
t = torch.frombuffer(t_mem, dtype=torch.float)

# Will be very slow if the system doesn't have enough RAM free
with open("st_datatape.f32", "wb") as f:
	for i in range(30522):
		t[:] = sorted.T[i]
		f.write(t_mem)
		print(i, "/ 30522")

print("Transposed")

with open("idx.json", "r") as f:
	idx = json.load(f)

titles = list(idx.keys())
remap_idx = {}

for i, j in enumerate(indices.tolist()):
	title = titles[j]
	remap_idx[title] = idx[title]

with open("idx_remap.json", "w") as f:
	json.dump(remap_idx, f)
