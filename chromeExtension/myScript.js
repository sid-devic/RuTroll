

{/* <script src="bundle.js"></script> */}
{/* <script type="text/javascript" src="auth.json"></script> */}
// $.getJSON("auth.json", function(json) {
//     console.log(json); // this will show the info it in firebug console
// });



// const scopes = 'https://www.googleapis.com/auth/cloud-platform'
// const jwt = new google.auth.JWT(key.client_email, null, key.private_key, scopes);


var tweets = document.getElementsByClassName('js-tweet-text tweet-text');
var message = document.getElementsByClassName('js-tweet-text tweet-text');
var header = document.getElementsByClassName('FullNameGroup');
var icon = document.getElementsByClassName('avatar js-action-profile-avatar');

var endpoint = "https://automl.googleapis.com/v1beta1/projects/tidy-arcade-220013/locations/us-central1/models/TCN2618236018595798875:predict"

predictTweet = (i, tweetText) => {
  fetch(endpoint, {
    method: 'POST',
    body: JSON.stringify({
        "payload" : {
          "textSnippet": {
               "content": tweetText,
                "mime_type": "text/plain"
           },
        }
      }),
    headers: {
      "Content-type": "application/json; charset=UTF-8",
      "Authorization": "Bearer ya29.c.Elo9BhsVNqmCwRGlEvx81RXtVrJa73jCcV3ggLzX44nXdkvvOWFIJhrQKrAl4gwRdVhgkiUambxlwtad0XRX_ekw_nlVb4qC1yxnxDev56ntBRBVUQVdcjT8swo"
    }
  })
  .then(response => response.json())
  .then(json => {
    console.log(tweetText, json['payload']);
    if (json['payload'][0]['displayName'] == 'russian') {
      
      header[i].innerText = "RUSSIAN HACKER";
      header[i].style.color = 'red';
      header[i].style.fontWeight="bold"
      message[i].style.backgroundColor = "yellow";
      icon[i].src = "https://static.thenounproject.com/png/461894-200.png"
      // tweets[i].innerText = json['payload'][0]['classification']['score'];
    }
  }
  )
}

for (var i = 0, l = message.length; i < l; i++) {
    predictTweet(i, message[i].innerText)
}
