var log = console.log

// We have the following structure
// {
//	"elements": { "word1": 1.45, "word2": 0.30, ... },
//	"magnitude": 70.5
// }
function dot(a, b) {
	var sum = 0.0

	// Try to select the shorter sequence for the loop
	var shorter = b.elements
	var longer = a.elements

	// Swap if we guessed wrong
	if (a.magnitude < b.magnitude) {
		shorter = a.elements
		longer = b.elements
	}

	for (var word in shorter) {
		if (word in longer) {
			sum += shorter[word] * longer[word]
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
