import spotipy
import os
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Load client-id and secret
load_dotenv()

# Spotify client-id
cid = os.getenv("SPOTIPY_CLIENT_ID")
secret = os.getenv("SPOTIPY_CLIENT_SECRET")

# Set up Spotify client credentials manager
client_credentials_manager = SpotifyClientCredentials(
    client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Christmas playlist
playlist1 = "73IgCB7ewVg3y8S3B8Pvnn"

# Midsummer playlist
playlist2 = "5fb1KO0EhFdZWwQsvJSIdM"


def playlist_tracks(playlist_id):  # Get list of tracks from the playlists
    tracks_list = []
    results = sp.playlist_tracks(playlist_id, fields="items(track)")
    tracks = results['items']
    tracks_list += [item['track'] for item in tracks]
    return tracks_list


def playlist_URIs(playlist_id):  # Get list of tracks from the URIs
    return [t["uri"] for t in playlist_tracks(playlist_id)]


def audio_features(track_URIs):  # Get audio features
    features = []
    r = splitlist(track_URIs, 5)
    for pack in range(len(r)):
        features = features + (sp.audio_features(r[pack]))
    df = pd.DataFrame.from_dict(features)
    df["uri"] = track_URIs
    return df


def splitlist(track_URIs, step):  # Split into sublists
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
audio_features1["target"] = 1
audio_features2["target"] = 0

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
pca = PCA(n_components=10)
pca_result = pca.fit_transform(features_pca_scaled)

# Accessing loadings
loadings = pca.components_

# Identify the best principal component based on variance explained
best_component_index = np.argmax(pca.explained_variance_ratio_)

# Get the loadings and feature names for the best component
best_loadings = loadings[best_component_index]
best_feature_names = features_pca.columns

# Sort features based on importance
sorted_indices = np.argsort(np.abs(best_loadings))[::-1]

# Get the feature names based on the importance order
feature_names = best_feature_names[sorted_indices]
print(feature_names)

# Print loadings for the best component
print(
    f"Best Principal Component Loadings (Component {best_component_index + 1}):")
for feature, loading in zip(feature_names, best_loadings[sorted_indices]):
    print(f"{feature}: {loading}")

# Plot the most important features
plt.bar(feature_names, np.abs(
    best_loadings[sorted_indices]), color='blue')
plt.xlabel('Features')
plt.ylabel('Absolute Loadings')
plt.title('Most Important Features')
# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')
plt.show()

# Chosen features based on PCA
feature_selection = ['energy', 'loudness', 'acousticness', 'danceability']

# Subset data for each playlist
christmas_songs = training_data[training_data['target'] == 1]
midsummer_songs = training_data[training_data['target'] == 0]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
songs_datasets = [christmas_songs, midsummer_songs]
colors = ['#D05555', '#7CE475']
labels = ['Christmas songs', 'Midsummer songs']

for i, feature in enumerate(feature_selection):
    row = i // 2
    col = i % 2

    # Plot histograms for each feature
    for dataset, color, label in zip(songs_datasets, colors, labels):
        _, bins = np.histogram(dataset[feature], bins=40)
        axes[row, col].hist(dataset[feature], bins=bins,
                            color=color, alpha=0.5, label=label)

    axes[row, col].set_title(feature, fontsize=12)
    axes[row, col].legend(loc='upper right')

plt.tight_layout()
plt.show()

# Grid search for hyperparameter tuning
X = training_data[feature_selection]
y = training_data['target']
cv = KFold(n_splits=10, random_state=0, shuffle=True)

# Random Forest
param_grid_rf = {'n_estimators': [50, 100, 200],
                 'max_depth': [None, 10, 20],
                 'min_samples_split': [2, 5, 10]}
grid_search_rf = GridSearchCV(RandomForestClassifier(
    random_state=0), param_grid_rf, cv=cv)
grid_search_rf.fit(X, y)
print("Best Random Forest Parameters:", grid_search_rf.best_params_)

# SVM
param_grid_svm = {'C': [0.1, 1, 10],
                  'kernel': ['linear', 'rbf']}
grid_search_svm = GridSearchCV(SVC(random_state=0), param_grid_svm, cv=cv)
grid_search_svm.fit(X, y)
print("Best SVM Parameters:", grid_search_svm.best_params_)


# Comparing Random Forest-and SVM classifier

# Split the data into features (X) and target variable (y)
X = training_data[feature_selection]
y = training_data['target']
cv = KFold(n_splits=10, random_state=0, shuffle=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100, random_state=0, min_samples_split=5, max_depth=10)
rf_cv_scores = cross_val_score(
    rf_classifier, X, y, cv=cv)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

# SVM Classifier
svm_classifier = SVC(kernel='linear', C=0.1)
svm_cv_scores = cross_val_score(
    svm_classifier, X, y, cv=cv)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

# Plotting the CV scores
plt.bar(['Random Forest', 'SVM'], [
        np.mean(rf_cv_scores), np.mean(svm_cv_scores)], color=['blue', 'green'])
plt.xlabel('Classifiers')
plt.ylabel('Mean CV Score')
plt.title('Cross-Validated Score Comparison')
plt.ylim(0, 1)
plt.show()

# Print result
print("Random Forest Classifier:")
print("Random Forest Cross-Validation Scores:", rf_cv_scores)
print("Mean Random Forest CV Score:", np.mean(rf_cv_scores))
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))

print("\nSVM Classifier:")
print("\nSVM Cross-Validation Scores:", svm_cv_scores)
print("Mean SVM CV Score:", np.mean(svm_cv_scores))
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Classification Report:\n", classification_report(y_test, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))


# Function to plot confusion matrix heatmap
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add values to the heatmap
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.5, str(cm[i, j]),
                     ha='center', va='center', color='red')
    plt.show()


# Plot confusion matrix for Random Forest
plot_confusion_matrix(y_test, rf_predictions, "Random Forest Classifier")

# Plot confusion matrix for SVM
plot_confusion_matrix(y_test, svm_predictions, "SVM Classifier")

# Evaluate the models
rf_accuracy = accuracy_score(y_test, rf_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# Plotting the accuracies
classifiers = ['Random Forest', 'SVM']
accuracies = [rf_accuracy, svm_accuracy]

plt.bar(classifiers, accuracies, color=['blue', 'green'])
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Classifier Comparison')
plt.ylim(0, 1)  # accuracy percentage
plt.show()
