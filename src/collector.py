import spotipy 
from spotipy.outh2 import SpotifyClientCredentials

sp = spotipy.Spotify(
    auth_manager = SpotifyClientCredentials(
        client_id = SPOTIFY_CLIENT_ID,
        client_secret = SPOTIFY_CLIENT_SECRET
    ),
    requests_timeout = 10,
    retries = 3
)

test = sp.search(q = "Radiohead", type = "artist", limit = 1)
print(f"API connection verified. Returns: {test['artists']['items'][0]['name']}")