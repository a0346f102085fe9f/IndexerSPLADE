from glob import glob
import json
import sys

# This script merges multiple sparse indices
# Takes a folder of folders of indices
# Produces a pair of extra large datatapes and a new idx.json
#

idx = {}
datatape_k = open("datatape_k.bin", "wb")
datatape_v = open("datatape_v.bin", "wb")

for folder in glob(sys.argv[1] + "/*"):
	with open(folder + "/idx.json", "rb") as f:
		idx.update(json.load(f))

	with open(folder + "/datatape_k.bin", "rb") as f:
		datatape_k.write(f.read())

	with open(folder + "/datatape_v.json", "rb") as f:
		datatape_v.write(f.read())

with open("idx.json", "w") as f:
	json.dump(idx, f, separators=(',', ':'))

datatape_k.close()
datatape_v.close()
