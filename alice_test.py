import csv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

playlist_uri = 'spotify:playlist:5fb1KO0EhFdZWwQsvJSIdM'  

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

results = spotify.playlist_tracks(playlist_uri)
tracks = results['items']
while results['next']:
    results = spotify.next(results)
    tracks.extend(results['items'])

# create a cvs file
with open('spotify_playlist.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Track']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for track in tracks:
        writer.writerow({'Track': track['track']['name']})