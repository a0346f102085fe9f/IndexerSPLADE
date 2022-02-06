# Derived from: https://github.com/naver/splade/blob/eb74d1ab31c42b9b94df7c0d67cc582b2c972dda/inference_SPLADE.ipynb
# Dependencies
import os
import sys
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


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
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}


# Split lots of tokens into 512-token slices
# 512 tokens is the limit of this model
def slice_tokenize(doc):
  tokenized_data = tokenizer(doc, return_tensors="pt")
  size = tokenized_data.input_ids.size()[1]
  slices = []
  offset = 0

  while offset < size:
    tgt = min(offset + 512, size)
    slice = {}
    slice['input_ids'] = tokenized_data.input_ids[:,offset:tgt].clone()
    slice['attention_mask'] = tokenized_data.attention_mask[:,offset:tgt]

    # Scrutiny paid off:
    # We saw a big drop in dimension count with initial slicing
    # Turns out that we were missing token 102, aka [SEP], that should go at the end
    # Adding it back forcibly brings the dimensions back up. The model appears to have heavy dependence on it being there.
    # Additionally, we are missing token 101 from the start of any consecutive slices, but it doesn't seem to make much difference
    # Be sure to execute .clone() first in order to avoid edition the original tokenized_data, since pytorch uses "views" for slicing
    if tgt - offset == 512:
      slice['input_ids'][0,511] = 102

    offset = offset + 511
    slices.append(slice)

  return slices


# Evaluate 512-token slices and add up the vectors each one produced
# Model returns sparse vectors of 30522 elements
def process_slice(data):
  with torch.no_grad():
      doc_rep = model(**data).squeeze()

  return doc_rep

def process_slices(slices):
  z = torch.zeros(30522)

  for slice in slices:
    z = z + process_slice(slice)

  # Precompute the magnitude
  # 1. Dot product self
  # 2. Take square root
  # 3. Tensor -> float
  mag = float(z.dot(z)**0.5)

  # get the number of non-zero dimensions in the rep:
  col = torch.nonzero(z).squeeze().cpu().tolist()

  # now let's inspect the bow representation:
  weights = z[col].cpu().tolist()
  d = {k: v for k, v in zip(col, weights)}
  sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
  elem = {}
  for k, v in sorted_d.items():
    elem[reverse_voc[k]] = v

  # Return JSON
  return { 'elements': elem, 'magnitude': mag }


# Main loop
path = sys.argv[1]
dir = os.listdir(path)
i = 0

print('var idx = {')

for filename in dir:
  file = open(path + filename, errors='ignore')
  text = file.read()
  file.close()
  slices = slice_tokenize(text)
  data = process_slices(slices)

  # Data has the following structure:
  # "file1.txt": {
  #   elements: { "word1": 1.50, "word2": 0.75, ... },
  #   magnitude: 70.8
  # }
  print("\"", filename, "\":", data, sep='')

  # Put a comma unless this is the last file
  if (len(dir) - i) > 1:
    print(",")

  i = i + 1
  sys.stderr.write(str(i) + "/" + str(len(dir)) + " " + filename + "\n")

print('}')
