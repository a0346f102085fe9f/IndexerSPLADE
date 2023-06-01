import torch

# >>> scores
# tensor([[0.00, 0.42, 0.06,  ..., 0.28, 0.49, 0.40],
#         [0.42, 0.00, 0.14,  ..., 0.35, 0.57, 0.47],
#         [0.06, 0.14, 0.00,  ..., 0.16, 0.08, 0.16],
#         ...,
#         [0.28, 0.35, 0.16,  ..., 0.00, 0.35, 0.37],
#         [0.49, 0.57, 0.08,  ..., 0.35, 0.00, 0.47],
#         [0.40, 0.47, 0.16,  ..., 0.37, 0.47, 0.00]])
torch.set_printoptions(precision=2, sci_mode=False)

N = 30000

# RAM cost: 3.41G
with open("datatape.f32", "rb") as f:
	b_datatape = f.read(N*30522*4)

x = torch.frombuffer(b_datatape, dtype=torch.float).view([N, 30522])

# A hefty matmul that may take some minutes to complete
# RAM cost: 3.35G
scores = x @ x.T

# Cosine similarity
# RAM cost: free (??)
mags = scores.diag()**0.5
scores /= mags.outer(mags)

# Mask self-similarity away so it is fair
diag_mask = torch.eye(N, dtype=torch.bool)
scores[diag_mask] = 0.0

# Find the single highest scoring document
# The matrix is mirrored so there is no distinction between rows/columns really
highest_score_doc = scores.sum(0).argmax()

print("Highest scoring document:", highest_score_doc)

# Save it a row at a time
with open("similarity_scores.f32", "wb") as f:
	buf = bytes(N*4)
	dst = torch.frombuffer(buf, dtype=torch.float)

	for src in scores:
		dst[:] = src
		f.write(buf)
