

{/* <script src="bundle.js"></script> */}
{/* <script type="text/javascript" src="auth.json"></script> */}
// $.getJSON("auth.json", function(json) {
//     console.log(json); // this will show the info it in firebug console
// });



// const scopes = 'https://www.googleapis.com/auth/cloud-platform'
// const jwt = new google.auth.JWT(key.client_email, null, key.private_key, scopes);


var tweets = document.getElementsByClassName('js-tweet-text tweet-text');
for (var i = 0, l = tweets.length; i < l; i++) {
    tweets[i].innerText = 'Some text';
}

const endpoint = "https://automl.googleapis.com/v1beta1/projects/tidy-arcade-220013/locations/us-central1/models/TCN2618236018595798875:predict"
fetch(endpoint, {
    method: 'POST',
    body: JSON.stringify({
        "payload" : {
          "textSnippet": {
               "content": "I love playing with my cat!",
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
  .then(json => console.log(json))

