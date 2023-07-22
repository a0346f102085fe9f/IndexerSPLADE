import zipfile
import json
import io

def load_history():
	try:
		file = open("history.json", "r")
	except FileNotFoundError:
		save_history({})
		return {}

	data = json.load(file)
	file.close()

	return data

def save_history(data):
	file = open("history.json", "w")
	json.dump(data, file)
	file.close()

history = load_history()
master = zipfile.ZipFile("20230601.zip", "r")

# Fuck you and your BeatifulSoup
# Extract all innerText
# Takes b"" strings
def process_html(html):
	# Text will likely follow a tag closure, but not always
	tails = html.split(b">")
	result = b""
	
	# HTML scope tracking
	# ==============================================================================
	stack = []

	# SPECIAL HANDLING
	# CAUSE: epub/r/redblue293-42736/a_chaotic_conundrum-116592.epub/Chapter5.html
	# An instance of "<error>" occurs, probably the user put it there
	def open_tag(tagname):
		if tagname in [b"link", b"meta", b"img", b"hr", b"br", b"iframe", b"error"]:
			return

		stack.append(tagname)

	# SPECIAL HANDLING
	# CAUSE: epub/f/fusion_fool_the_3rd-16866/twilights_reasons_to_hate_fusion_fool_the_mini_series-103969.epub/Chapter6.html [What a spicy document]
	# An instance of "</hr>" occurs, which isn't valid HTML?
	def close_tag(tagname):
		if tagname in [b"hr"]:
			return

		if stack[-1] != tagname:
			raise Exception("Trying to close", tagname, "when in", stack[-1])

		stack.pop()
	# ==============================================================================

	for tail in tails:
		# tail.startswith(b"</")	--> Tag closed, no innerText
		# tail.startswith(b"<")		--> Tag opened, no innerText
		# tail.find(b"</") > 0		--> Tag closed, preceded by innerText
		# tail.find(b"<") > 0		--> Tag opened, preceded by innerText
		if tail.startswith(b"</"):
			insn = close_tag
			tagname = tail[2:]
			innerText = None
		elif tail.startswith(b"<"):
			insn = open_tag
			tagname = tail[1:].split(b" ")[0]
			innerText = None
		elif tail.find(b"</") >= 0:
			insn = close_tag
			offset = tail.find(b"</")
			innerText = tail[:offset]
			tagname = tail[offset:][2:]
		elif tail.find(b"<") >= 0:
			insn = open_tag
			offset = tail.find(b"<")
			innerText = tail[:offset]
			tagname = tail[offset:][1:].split(b" ")[0]
		else:
			raise Exception("Invalid HTML")

		# Track the scope
		insn(tagname)

		# Stop on </html>
		if insn == close_tag and tagname == b"html":
			return result

		# Don't care about <title>
		if tagname in [b"title"]:
			continue

		# Stuff gets a little involved...
		# Skip indenting
		if innerText:
			if innerText.startswith(b"\n") and innerText.strip() == b"":
				pass
			else:
				result += innerText

		# Reconstruct vertical spacing
		if tagname == b"br":
			result += b"\n"

		if insn == open_tag:
			continue

		if tagname in [b"p", b"h1", b"h2", b"h3", b"h4", b"h5", b"h6"]:
			result += b"\n\n"

	raise Exception("Document ends before </html> is encountered")

def process_epub(path, file):
	epub = zipfile.ZipFile(io.BytesIO(file.read(path)), "r")
	title = path.split("/")[-1][:-5] + ".txt"
	file = open("txt/" + title, "wb")

	print(title)

	# There are some weird chapter names, like "Chapter26_split_000.html"
	# Sometimes they come out of order: "Chapter1.html" is followed by "Chapter10.html"
	# Filelist is not sorted by default
	# We will have to do it manually
	chapters = []

	for entry in epub.filelist:
		if entry.filename.startswith("Chapter"):
			chapters.append(entry.filename)
	
	chapters.sort()

	for filename in chapters:		
		# Gives raw bytes
		# Do not reparse
		chapter = epub.read(filename)
		file.write(process_html(chapter))

	file.close()



processed = 0
skipped = 0

for entry in master.filelist:
	path = entry.filename
	if path.startswith("epub"):
		if path in history:
			skipped += 1
		else:
			process_epub(path, master)
			history[path] = 1
			processed += 1
	if processed >= 30000:
		break

print("Processed", processed, "entries, skipped", skipped, "entries.")
save_history(history)
print("History file updated.")
