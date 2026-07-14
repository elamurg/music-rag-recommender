"""FastAPI with three endpoints:
POST /recommend - accepts JSON body with string query, returns ranked rec with justification and Spotify URL
GET /health - returns 200 OK with basic system status
GET /stats - returns corpus statistics"""