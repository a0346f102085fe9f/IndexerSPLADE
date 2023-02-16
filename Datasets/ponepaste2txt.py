import tarfile
import csv
import os

# Python's tarfile is hilariously slow
# But ok
tar = tarfile.open("latest.tar.gz")
tar.extractall("txt/")

prefix = "txt/home/debian/dumps/14-02-2023"

idx_file = open(f"{prefix}/pastes.csv", "rb")
idx = list(csv.reader(idx_file.read().decode().split("\n")))
idx_file.close()

for entry in idx:
	if entry == []:
		exit()

	paste_id, title, uploaded_at, edited_at, author = entry
	title = title.replace("/", "_").replace(":", "").replace("?", "").replace("*", "").replace("|", "")
	filename = f"txt/{title}-{paste_id}.txt"

	print(filename)

	os.rename(f"{prefix}/data/{paste_id}", filename)

	"""
	file = open(filename, "wb")
	file.write(master.extractfile().read())
	file.close()
	"""
