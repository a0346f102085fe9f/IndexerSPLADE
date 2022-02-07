var log = console.log


// Datatape K 	Uint16 token IDs
// Datatape V 	Float32 token weights
var datatape_k
var datatape_v
var idx

function populate_views() {
	log("Populating datatape views...")

	var time_start = performance.now()

	// Offset is in bytes
	var tape_offset_k = 0
	var tape_offset_v = 0

	for (var filename in idx) {
		var entry = idx[filename]

		var keys = new Uint16Array(datatape_k, tape_offset_k, entry.dimensions)
		var values = new Float32Array(datatape_v, tape_offset_v, entry.dimensions)
		
		var expkeys = new Uint16Array(30522)
		var i = 0

		expkeys.fill(-1)

		for (var key of keys) {
			expkeys[key] = i
			i++
		}

		entry.keys = keys
		entry.values = values
		entry.expkeys = expkeys

		tape_offset_k += entry.dimensions * 2
		tape_offset_v += entry.dimensions * 4
	}

	var time_done = performance.now()

	log("Done! Took", time_done - time_start, "ms")
}


// We have the following structure
// {
//	"elements": 1420,
//	"magnitude": 70.5,
//  "keys": [ Uint16 ],
//  "values": [ Float32 ]
// }
function dot(a, b) {
	var sum = 0.0

	// Try to select the shorter sequence for the loop
	var shorter = b
	var longer = a

	// Swap if we guessed wrong
	if (a.dimensions < b.dimensions) {
		shorter = a
		longer = b
	}

	var sk = shorter.keys
	var sv = shorter.values
	var lk = longer.expkeys
	var lv = longer.values

	for (var idx_s in sk) {
		var key = sk[idx_s]
		var idx_l = lk[key]

		if (idx_l <= 30522) {
			sum += sv[idx_s] * lv[idx_l]
		}
	}

	return sum
}

// Cosine similarity is absolutely necessary for SPLADE
// It just doesn't work at all if you only do the dot()
function cosine_similarity(a, b) {
	return dot(a, b) / (a.magnitude * b.magnitude)
}


// We have the following structure
// {
//	"file1.txt": {},
//	"file2.txt": {},
// }
function find_similar_to(title) {
	var results = []
	var a = idx[title]

	for (var target in idx) {
		if (target === title)
			continue

		var b = idx[target]
		var ab = cosine_similarity(a, b)
		
		results.push( { title: target, score: ab } )
	}

	var sort_fn = function(a, b) { return b.score - a.score }

	results.sort(sort_fn)
	log(results)
}
