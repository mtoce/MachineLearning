import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import json

def load_and_clean():
    """
    spotify, identify = load_and_clean()
    """
    spotify = pd.read_csv('SpotifyFeatures.csv')

    # dataframe that serves to identify songs
    identify = spotify[['artist_name', 'track_id', 'track_name']]

    # dataframe consisting of audio features we want to train on
    spotify = spotify.drop(columns = ['genre',
                                    'mode',
                                    'time_signature',
                                    'key',
                                    'track_id',
                                    'artist_name',
                                    'popularity',
                                    'track_name',
                                    'duration_ms',
                                    'speechiness'])

    return spotify, identify

spotify, identify = load_and_clean()

def knn_predictor(audio_feats, k=20):
    """
    similar_song_ids, visual_df = knn_predictor(audio_features)
    """
    # Scale the data with standard scaler
    scaler = StandardScaler()
    spotify_scaled = scaler.fit_transform(spotify)

    ################################################
    audio_feats_scaled = scaler.transform([audio_feats])

    ## Nearest Neighbors model
    knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    knn.fit(spotify_scaled)

    # pickle the model for later use
    filename = 'knn_model.sav'
    pickle.dump(knn, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))
    # make prediction 
    prediction = loaded_model.kneighbors(audio_feats_scaled)

    # create an index for similar songs
    similar_songs_index = prediction[1][0][:k].tolist()
    
    # Create an empty list to store simlar song names
    similar_song_ids = []
    similar_song_names = []

    # loop over the indexes and append song names to empty list above
    for i in similar_songs_index:
        song_id = identify['track_id'].iloc[i]
        similar_song_ids.append(song_id)
        song_name = identify['track_name'].iloc[i]
        similar_song_names.append(song_name)

    #################################################

    column_names = spotify.columns.tolist()

    # put scaled audio features into a dataframe
    audio_feats_scaled_df = pd.DataFrame(audio_feats_scaled, columns=column_names)

    # create empty list of similar songs' features
    similar_songs_features = []

    # loop through the indexes of similar songs to get audio features for each
    #. similar song
    for index in similar_songs_index:
        list_of_feats = spotify.iloc[index].tolist()
        similar_songs_features.append(list_of_feats)

    # scale the features and turn them into a dataframe
    similar_feats_scaled = scaler.transform(similar_songs_features)
    similar_feats_scaled_df = pd.DataFrame(similar_feats_scaled, columns=column_names)

    # get the % difference between the outputs and input songs
    col_names = similar_feats_scaled_df.columns.to_list()
    diff_df = pd.DataFrame(columns=col_names)
    for i in range(k):
        diff = abs(similar_feats_scaled_df.iloc[i] - audio_feats_scaled_df.iloc[0])
        # print('type: ', type(similar_feats_scaled_df.iloc[i]))
        diff_df.loc[i] = diff
    
    # add sums of differences 
    diff_df['sum'] = diff_df.sum(axis=1)
    diff_df = diff_df.sort_values(by=['sum'])
    diff_df = diff_df.reset_index(drop=True)

    # add track_id to DF
    diff_df['track_id'] = similar_song_ids

    # reorder cols to have track_id as first column
    cols = list(diff_df)
    cols.insert(0, cols.pop(cols.index('track_id')))
    diff_df = diff_df.loc[:, cols]

    # Grab only the unique 10 songs
    diff_df = diff_df.drop_duplicates(subset=['sum'])[:10]

    return diff_df

test_audio_features = [0.5, 0.5, 0.5, 0.1, 0.25, -5.0, 125, 0.5]
diff_df = knn_predictor(test_audio_features)

diff_json = diff_df.to_json(orient='records')


print(diff_df)