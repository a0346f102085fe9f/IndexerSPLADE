# Derived from: https://github.com/naver/splade/blob/eb74d1ab31c42b9b94df7c0d67cc582b2c972dda/inference_SPLADE.ipynb
# Dependencies
import os
import sys
import torch
from array import array
from transformers import AutoModelForMaskedLM, AutoTokenizer


# Use GPU if running on Google Colab
# The model isn't large, it will run on anything
# Even a K80 gives a massive speed boost over a Ryzen 3700X
# Use !nvidia-smi to check what card you got
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Model
# Extends torch module
# Overrides `forward()`
class Splade(torch.nn.Module):

    def __init__(self, model_type_or_dir, agg="max"):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        assert agg in ("sum", "max")
        self.agg = agg

    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        if self.agg == "max":
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
            return values
            # 0 masking also works with max because all activations are positive
        else:
            return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)


# Model folder
#
#### v1
# agg = "sum"
# model_type_or_dir = "weights/flops_efficient"
# model_type_or_dir = "weights/flops_best"

##### v2
agg = "max"
model_type_or_dir = "weights/splade_distil_CoCodenser_large"
# model_type_or_dir = "weights/splade_max"
# model_type_or_dir = "weights/distilsplade_max"
# Model taken from http://download-de.europe.naverlabs.com/Splade_Release_Jan22/splade_distil_CoCodenser_large.tar.gz


# Model loading
#
model = Splade(model_type_or_dir, agg=agg)
model.eval()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}


# Tokenize
#
# Tokenizer produces this structure:
# { 'input_ids': [], 'token_type_ids': [], 'attention_mask': [] }
#
# We are not interested in 'token_type_ids'.
#
# Split lots of tokens into 512-token slices
# 512 tokens is the limit of this model
def slice_tokenize(doc):
	slices = []
	slice = [[],[]]

	slice_size = 0
	total_size = 0

	for line in doc.split("\n"):
		line = line.strip()

		if line == "":
			continue

		tokens = tokenizer(line)

		size = len(tokens['input_ids'])

		if (size > 512):
			print("ABORT: Encountered a very long line of text. Is it minified code? Stray binary data?")
			return 0

		if (slice_size + size > 512):
			slices.append( {'input_ids': torch.tensor([slice[0]]).to(device), 'attention_mask': torch.tensor([slice[1]]).to(device) } )
			slice = [[],[]]
			slice_size = 0
			# Flush

		slice[0].extend( tokens['input_ids'] )
		slice[1].extend( tokens['attention_mask'] )
		slice_size = slice_size + size
		total_size = total_size + size

	if (slice_size > 0):
		slices.append( {'input_ids': torch.tensor([slice[0]]).to(device), 'attention_mask': torch.tensor([slice[1]]).to(device) } )
		slice = [[],[]]
		slice_size = 0
		# Flush

	return slices


# Evaluate 512-token slices and add up the vectors each one produced
# Model returns sparse vectors of 30522 elements
def process_slice(data):
  with torch.no_grad():
      doc_rep = model(**data).squeeze()

  return doc_rep

def process_slices(slices):
  z = torch.zeros(30522).to(device)

  for slice in slices:
    z = z + process_slice(slice)

  # Precompute the magnitude
  # 1. Dot product self
  # 2. Take square root
  # 3. Tensor -> float
  mag = float(z.dot(z)**0.5)

  # get the number of non-zero dimensions in the rep:
  col = torch.nonzero(z).squeeze().cpu().tolist()

  # Sort the elements
  # Those that have the highest weight come first
  weights = z[col].cpu().tolist()
  d = {k: v for k, v in zip(col, weights)}
  sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

  return (sorted_d, mag)


# Main loop
path = sys.argv[1]
dir = os.listdir(path)
i = 0

# We'll produce three files:
# - idx.json
# - datatape_k.bin
# - datatape_v.bin
#
# JSON has the following structure:
# "file1.txt": {
#   dimensions: 1420,
#   magnitude: 70.8
# }
#
# Datatapes are:
# datatape_k.bin  UInt16 data
# datatape_v.bin  Float32 data
#
#

datatape_k = open("datatape_k.bin", "wb")
datatape_v = open("datatape_v.bin", "wb")
idx = open("idx.json", "w", encoding='utf-8', errors='replace')

idx.write("{")

for filename in dir:
  file = open(path + filename, encoding='utf-8', errors='replace')
  text = file.read()
  file.close()
  slices = slice_tokenize(text)
  if slices:
    data, mag = process_slices(slices)
  else:
    data = {}
    mag = 0.0

  k_array = array("h", data.keys())
  v_array = array("f", data.values())

  k_array.tofile(datatape_k)
  v_array.tofile(datatape_v)

  json = "\"" + filename + "\":{\"dimensions\":" + str(len(data)) + ",\"magnitude\":" + str(mag) + "}"
  idx.write(json)

  # Put a comma unless this is the last file
  if (len(dir) - i) > 1:
    idx.write(",")

  i = i + 1
  print(str(i) + "/" + str(len(dir)) + " " + filename)

idx.write("}")

datatape_k.close()
datatape_v.close()
idx.close()
