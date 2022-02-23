#
# DO NOT USE
#

# Derived from: https://github.com/naver/splade/blob/eb74d1ab31c42b9b94df7c0d67cc582b2c972dda/inference_SPLADE.ipynb
# Dependencies
import sys
import json
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


# example document from MS MARCO passage collection (doc_id = 8003157)
# doc = "Glass and Thermal Stress. Thermal Stress is created when one area of a glass pane gets hotter than an adjacent area. If the stress is too great then the glass will crack. The stress level at which the glass will break is governed by several factors."

# Use argv instead
doc=sys.argv[1]


# Tokenize
#
# Tokenizer produces this structure:
# { 'input_ids': [], 'token_type_ids': [], 'attention_mask': [] }
#
# We are not interested in 'token_type_ids'.
#
# Split lots of tokens into 512-token slices
# 512 tokens is the limit of this model
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
		exit()

	if (slice_size + size > 512):
		slices.append( {'input_ids': torch.tensor([slice[0]]), 'attention_mask': torch.tensor([slice[1]]) } )
		slice = [[],[]]
		slice_size = 0
		# Flush

	slice[0].extend( tokens['input_ids'] )
	slice[1].extend( tokens['attention_mask'] )
	slice_size = slice_size + size
	total_size = total_size + size

if (slice_size > 0):
	slices.append( {'input_ids': torch.tensor([slice[0]]), 'attention_mask': torch.tensor([slice[1]]) } )
	slice = [[],[]]
	slice_size = 0
	# Flush

print("Tokens in total: ", total_size)
print(slices)


# Print tokenized input so you can see what the model sees
# ['[CLS]', 'word1', 'word2', 'word3', ..., '[SEP]']
def dump_slice(slice):
  token_ids = slice['input_ids'].tolist()[0]
  tokens = []
  for id in token_ids:
    tokens.append(reverse_voc[id])

  print(tokens)

for slice in slices:
  dump_slice(slice)


# Evaluate 512-token slices and add up the vectors each one produced
# Model returns sparse vectors of 30522 elements
def process_slice(data):
  with torch.no_grad():
      doc_rep = model(**data).squeeze()

  #print("Slice!")
  return doc_rep

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
print("number of actual dimensions: ", len(col))

# now let's inspect the bow representation:
weights = z[col].cpu().tolist()
d = {k: v for k, v in zip(col, weights)}
sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
elem = {}
for k, v in sorted_d.items():
    elem[reverse_voc[k]] = round(v, 2)

_json = { 'elements': elem, 'magnitude': mag }


print("JSON representation:")
print(json.dumps(_json, indent=4))
