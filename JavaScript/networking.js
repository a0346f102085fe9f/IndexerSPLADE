function get(address, callback_ready, type = "text") {
	var xhr = new XMLHttpRequest();

	xhr.onload = async function() {
		callback_ready(xhr.response)
	}

	xhr.onprogress = function(event) {
		log("Downloading [" + address + "]: " + (event.loaded / event.total * 100) + "% " + Math.round(event.loaded / 1024) + "KB")
	}

	xhr.onerror = function(event) {
		error(locale.network_error)
	}

	xhr.open('GET', address, true);
	//xhr.overrideMimeType("application/json");
	xhr.responseType = type
	xhr.send(null);
}
