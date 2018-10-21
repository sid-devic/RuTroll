const endpoint = "https://automl.googleapis.com/v1beta1/projects/tidy-arcade-220013/locations/us-central1/models/TCN5448917125241496931:predict"
const payload = {
    "payload" : {
        "textSnippet": {
           "content": "I love playing with my cat!",
            "mime_type": "text/plain"
        },
    }
}


var request = new XMLHttpRequest();

request.onreadystatechange = function() {
    if (request.readyState == 4 && request.status == 200) {
        console.log(request.responseText);
    }
    else {
        console.log("Failed: "+request.responseText);
    }
}

request.open('POST', endpoint, true);
request.send(JSON.stringify(payload));

var tweets = document.getElementsByClassName('js-tweet-text tweet-text');
for (var i = 0, l = tweets.length; i < l; i++) {
    tweets[i].innerText = 'Some text';
}
