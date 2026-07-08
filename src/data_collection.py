# to test if the last.fm and genius api connections work
import os
from dotenv import load_dotenv, find_dotenv
import pylast
import lyricsgenius

dotenv_path = find_dotenv()
print(f"dotenv found at: {dotenv_path!r}")
loaded = load_dotenv(dotenv_path)
print(f"load_dotenv returned: {loaded}")
print(f"LASTFM_API_KEY: {os.getenv('LASTFM_API_KEY')!r}")
print(f"LASTFM_API_SECRET: {os.getenv('LASTFM_API_SECRET')!r}")
print(f"GENIUS_ACCESS_TOKEN: {os.getenv('GENIUS_ACCESS_TOKEN')!r}")

lastfm_api_key = os.getenv("LASTFM_API_KEY")
lastfm_api_secret = os.getenv("LASTFM_API_SECRET")
genius_access_token = os.getenv("GENIUS_ACCESS_TOKEN")

if not lastfm_api_key:
    raise EnvironmentError("Last.fm API key not found.")
if not genius_access_token:
    raise EnvironmentError("Genius access token not found.")

# client objects
lastfm = pylast.LastFMNetwork(
    api_key=lastfm_api_key,
    api_secret=lastfm_api_secret or "",
)

genius = lyricsgenius.Genius(genius_access_token)
genius.verbose = False
genius.remove_section_headers = True
genius.timeout = 15

print("Test 1: find track")
# parameters: search_for_track(artist, track) = passing empty string for artist
SEARCH_QUERY = "Blinding Lights"
track_search = lastfm.search_for_track("", SEARCH_QUERY)
top_results = track_search.get_next_page()
if not top_results:
    raise ValueError(f"No results for '{SEARCH_QUERY}'")
track = top_results[0]
track_name = track.get_name()
artist_name = track.get_artist().get_name()
# mbid is the musicbrainz identifier = a stable canonical uuid for the recording
track_mbid = track.get_mbid()
print(f"Track Name: {track_name}")
print(f"Artist: {artist_name}")
print(f"MBID: {track_mbid}")

print("Test 2: text features (replaces audio features)")
# spotify's audio features were 12 floats (danceability, energy, valence, etc).
# they don't exist in the last.fm world. instead we get richer natural language:
# - tags: crowdsourced descriptive labels with weights (mood, genre, era, vibe)
# - listener/playcount stats: numeric popularity signal
# - wiki summary: short editorial paragraph
# - lyrics (from genius): the song's actual words
# for a RAG system this is strictly more useful — text embeds natively into FAISS
# and the LLM can reason over it without any feature-engineering step

print("Top tags:")
# get_top_tags returns TopItem objects: .item is the Tag, .weight is 0-100 popularity
top_tags = track.get_top_tags(limit=10)
for tag in top_tags:
    print(f"  {tag.item.get_name()}: weight {tag.weight}")

print(f"Listener count: {track.get_listener_count()}")
print(f"Playcount: {track.get_playcount()}")

# wiki_summary returns a short editorial paragraph or empty string
wiki_summary = track.get_wiki_summary()
if wiki_summary:
    preview = wiki_summary[:300].replace("\n", " ")
    print(f"Wiki summary: {preview}...")
else:
    print("No wiki summary available")

# lyrics from genius, search_song returns a Song object or None
print("Lyrics from Genius (first 300 chars):")
genius_song = genius.search_song(track_name, artist_name)
if genius_song and genius_song.lyrics:
    preview = genius_song.lyrics[:300].replace("\n", " ")
    print(f"{preview}...")
else:
    print("Lyrics not found")

print("Test 3: similar tracks (replaces recommendations)")
# get_similar returns SimilarItem objects with .item (Track) and .match (0-1).
# last.fm computes similarity from co-listening patterns across all its users,
# conceptually equivalent to the collaborative-filtering signal that spotify's
# deprecated recommendations endpoint used to provide
similar_tracks = track.get_similar(limit=5)
print("Similar tracks:")
for i, similar in enumerate(similar_tracks):
    similar_track = similar.item
    similarity = similar.match
    similar_artist = similar_track.get_artist().get_name()
    print(f"{i+1}. {similar_track.get_name()} by {similar_artist} (similarity: {similarity:.3f})")

print("API connections work.")