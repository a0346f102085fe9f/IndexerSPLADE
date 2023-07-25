import torch

# This script will perform some reversible filtering on the 30000x30522 float matrix produced sparse2dense.py
# 1. The matrix is transposed, which results in more consecutive zeros and makes gzip a little happier
# 2. The rows are sorted such that the rows that have the most nonzero elements come first
#

with open("datatape.f32", "rb") as f:
	b_src = bytearray(f.read())

assert len(b_src) % (4*30522) == 0

n = len(b_src) // (4*30522)
b_dst = bytearray(n*30522*4)

x = torch.frombuffer(b_src, dtype=torch.float).view([n, 30522])
t = torch.frombuffer(b_dst, dtype=torch.float).view([30522, n])
t[:] = x.T

with open("t_datatape.f32", "wb") as f:
	f.write(b_dst)

x = None
b_src = None

nz_counts = torch.tensor([row.count_nonzero() for row in t])
s_nz_count, indices = nz_counts.sort(descending=True)

t[:] = t[indices]

with open("ts_datatape.f32", "wb") as f:
	f.write(b_dst)
