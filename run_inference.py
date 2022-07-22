# Derived from: https://github.com/naver/splade/blob/eb74d1ab31c42b9b94df7c0d67cc582b2c972dda/inference_SPLADE.ipynb
# Dependencies
import os
import sys
import json
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

def inspect_json(sorted_d, mag):
  elem = {}
  for k, v in sorted_d.items():
    elem[reverse_voc[k]] = round(v, 2)

  _json = { 'elements': elem, 'magnitude': mag }

  print("JSON representation:")
  print(json.dumps(_json, indent=4))


# Split lots of tokens into 512-token slices
# 512 tokens is the limit of this model
def slices(tokenized_data):
  input_ids = tokenized_data.input_ids
  attention_mask = tokenized_data.attention_mask
  size = len(input_ids)
  slices = []

  # Scrutiny paid off:
  # We saw a big drop in dimension count with initial slicing
  # Turns out that we were missing token 102, aka [SEP], that should go at the end
  # Adding it back forcibly brings the dimensions back up. The model appears to have heavy dependence on it being there.
  # Additionally, we are missing token 101 from the start of any consecutive slices, but it doesn't seem to make much difference
  for i in range(1, 1 + size // 512):
    input_ids.insert(i * 512 - 1, 102)
    attention_mask.insert(i * 512 - 1, 1)

  while size > 0:
    t_input_ids = [ input_ids[:512] ]
    t_attention_mask = [ attention_mask[:512] ]

    slice = {}
    slice['input_ids'] = torch.tensor(t_input_ids).to(device)
    slice['attention_mask'] = torch.tensor(t_attention_mask).to(device)
    slices.append(slice)

    input_ids = input_ids[512:]
    attention_mask = attention_mask[512:]
    size -= 512

  return slices


# Evaluate 512-token slices and add up the vectors each one produced
# Model returns sparse vectors of 30522 elements
def process_slice(data):
  with torch.no_grad():
      doc_rep = model(**data).squeeze()

  return doc_rep

def process_tokenized(tokenized_data):
  z = torch.zeros(30522).to(device)

  for slice in slices(tokenized_data):
    z = z + process_slice(slice)

  # Precompute the magnitude
  # 1. Dot product self
  # 2. Take square root
  # 3. Tensor -> float
  mag = float(z.dot(z)**0.5)

  # get the number of non-zero dimensions in the rep:
  col = torch.nonzero(z).squeeze().cpu().tolist()

  # Build the compact vector
  weights = z[col].cpu().tolist()
  d = {}

  for k, v in zip(col, weights):
  	d[k] = v

  # Sort the elements
  # Those that have the highest weight come first
  sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

  # Mark stray tokens by negating their weight
  token_map = {}

  for k in tokenized_data.input_ids:
    token_map[k] = 1

  for k in sorted_d:
    if not k in token_map:
      sorted_d[k] = -sorted_d[k]

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

  tokenized_data = tokenizer(text)
  data, mag = process_tokenized(tokenized_data)

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
