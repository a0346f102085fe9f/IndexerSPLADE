from concurrent.futures import ThreadPoolExecutor
import gzip
import sys
import os

# This script compresses the rows of a transposed dense index
# Takes st_datatape.f32 from transpose.py
# Produces 30552 .gz files
#

file_size = os.stat(sys.argv[1]).st_size
row_size = file_size // 30522

# Force timestamp to 0 for deterministic results
# https://github.com/python/cpython/blob/main/Lib/gzip.py#L599
def compress_fn(data):
	return gzip.compress(data, mtime=0)

with open(sys.argv[1], "rb") as f, ThreadPoolExecutor(max_workers=4) as executor:
	futures = []

	while True:
		row = f.read(row_size)
		if not row:
			break

		futures.append(executor.submit(compress_fn, row))

	for i, future in enumerate(futures):
		result = future.result()
		print(i, len(result) / row_size)
		with open(f"results/{i}.gz", "wb") as out:
			out.write(result)
