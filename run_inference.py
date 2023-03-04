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


# Model folder or repo
#
#model_type_or_dir = "naver/neuclir22-splade-ru"
model_type_or_dir = "naver/splade-cocondenser-ensembledistil"


# Model loading
#
model = Splade(model_type_or_dir)
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


# Split the stack of 512-token slices further to avoid choking the GPU
def slices(tokenized_data):
  a_rect = tokenized_data['input_ids']
  b_rect = tokenized_data['attention_mask']

  # Apply batch size limit
  # Batch size should be increased if there's spare VRAM
  bsz = 4

  a_batch = a_rect.split(bsz)
  b_batch = b_rect.split(bsz)
  slices = []

  for a, b in zip(a_batch, b_batch):
    slice = {
      'input_ids': a.to(device),
      'attention_mask': b.to(device)
    }

    slices.append(slice)

  return slices

# Evaluate slices
# Data dimensions range from [1, 512] to [bsz, 512]
def process_slice(data):
  with torch.no_grad():
    doc_rep = model(**data)

  return doc_rep.cpu()

def process_tokenized(tokenized_data):
  reps = []

  for slice in slices(tokenized_data):
    reps.append(process_slice(slice))

  # Pool the embeddings
  # I have also tried max pooling instead of sum:
  # 1. It does not have much effect on search results
  # 2. It does have a lot of effect on auto tags (It ruins them)
  z = torch.vstack(reps).sum(0)

  # Precompute the magnitude
  # 1. Dot product self
  # 2. Take square root
  # 3. Tensor -> float
  mag = float(z.dot(z)**0.5)

  # get the number of non-zero dimensions in the rep:
  col = torch.nonzero(z).flatten().tolist()

  # Build the compact vector
  weights = z[col].tolist()
  d = {}

  for k, v in zip(col, weights):
  	d[k] = v

  # Sort the elements
  # Those that have the highest weight come first
  sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

  # Mark stray tokens by negating their weight
  token_set = set(tokenized_data.input_ids.flatten().tolist())

  for k in sorted_d:
    if not k in token_set:
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

  # This mix of parameters allows for long inputs
  # Resulting token_ids will look as follows:
  # [[101, ..., 102],
  #  [101, ..., 102],
  #  [101, ..., 102],
  #  [101, ..., 0]]
  #
  # A "stride" parameter is available to overlap slices by n tokens
  tokenized_data = tokenizer(text, truncation=True, padding=True, return_overflowing_tokens=True, return_tensors="pt")
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
