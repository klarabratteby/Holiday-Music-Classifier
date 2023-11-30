import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Spotify client-id
# Dodo: add to env-file for privacy
cid = '5d83fb4a8f6f4f8383a86ea0d6e56fb7'
secret = '8fe140a97f164d2e9b1b404bc44072d1'

# Set up Spotify client credentials manager
client_credentials_manager = SpotifyClientCredentials(
    client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Christmas playlist
playlist1 = "73IgCB7ewVg3y8S3B8Pvnn"

# Midsummer playlist
playlist2 = "5fb1KO0EhFdZWwQsvJSIdM"


def playlist_tracks(playlist_id):
    tracks_list = []
    results = sp.playlist_tracks(playlist_id, fields="items(track)")
    tracks = results['items']
    tracks_list += [item['track'] for item in tracks]
    return tracks_list


def playlist_URIs(playlist_id):
    return [t["uri"] for t in playlist_tracks(playlist_id)]

# Feature function


def audio_features(track_URIs):
    features = []
    r = splitlist(track_URIs, 5)
    for pack in range(len(r)):
        features = features + (sp.audio_features(r[pack]))
    df = pd.DataFrame.from_dict(features)
    df["uri"] = track_URIs
    return df


def splitlist(track_URIs, step):
    return [track_URIs[i::step] for i in range(step)]


list1 = playlist_URIs(playlist1)
list2 = playlist_URIs(playlist2)

# Dataframes
audio_features1 = audio_features(list1)
audio_features2 = audio_features(list2)

audio_features1.to_csv('christmas.csv')
audio_features2.to_csv('midsummer.csv')

# Label the data
# set label with true or false
audio_features1["target"] = 0
audio_features2["target"] = 1

training_data = pd.concat(
    [audio_features1, audio_features2], ignore_index=True)

# PCA

# Remove non-numeric columns if any
features_pca = training_data.drop(
    ['uri', 'type', 'id', 'track_href', 'analysis_url'], axis=1)
# print(features_pca.dtypes)

# Standardize the features
scaler = StandardScaler()
features_pca_scaled = scaler.fit_transform(features_pca)

# Apply PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(features_pca_scaled)

# Accessing loadings
loadings = pca.components_

# Identify the most important features based on absolute loadings
important_features = np.abs(loadings[2])  # Third principal component

# Sort features based on importance
sorted_indices = np.argsort(important_features)[::-1]

# Get the feature names based on the importance order
feature_names = features_pca.columns[sorted_indices]
print(feature_names)

# Plot the most important features
plt.bar(feature_names, important_features[sorted_indices], color='blue')
plt.xlabel('Features')
plt.ylabel('Absolute Loadings')
plt.title('Most Important Features (Third Principal Component)')
# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')
plt.show()
