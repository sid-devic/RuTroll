export GOOGLE_APPLICATION_CREDENTIALS= // PUT KEY FILE NAME HERE

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  -H "Content-Type: application/json" \
  https://automl.googleapis.com/v1beta1/projects/tidy-arcade-220013/locations/us-central1/models/TCN2618236018595798875:predict \
  -d '{
        "payload" : {
          "textSnippet": {
               "content": "what did you just say to me? I'll have you know I have made over 15000 api requests to AutoML in the past 15 hours.",
                "mime_type": "text/plain"
           },
        }
      }'