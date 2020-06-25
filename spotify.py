import pandas as pd
import numpy as np
import pickle
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import json
from sklearn.pipeline import make_pipeline

def load_and_clean():
    """
    spotify, identify = load_and_clean()
    """
    spotify = pd.read_csv('https://raw.githubusercontent.com/BW-pilot/MachineLearning/master/CSVs/spotify_final.csv')

    spotify = spotify.drop(columns=['Unnamed: 0'], axis=1)

    spotify.to_csv('spotify_final.csv', index=False)

    # dataframe that serves to identify songs
    identify = spotify[['artist_name', 'track_id', 'track_name']]

    # dataframe consisting of audio features we want to train on
    spotify = spotify.drop(columns = ['track_id',
                                    'artist_name',
                                    'track_name'])

    

    return spotify, identify

spotify, identify = load_and_clean()

def knn_predictor(audio_feats, k=100):
    """
    differences_df = knn_predictor(audio_features)
    """

    # Scale the data with standard scaler
    scaler = StandardScaler()
    spotify_scaled = scaler.fit_transform(spotify)

    ################################################
    audio_feats_scaled = scaler.transform([audio_feats])

    ## Nearest Neighbors model
    knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    knn.fit(spotify_scaled)

    # JOBLIB dump
    dump(knn, 'knn_final.joblib', compress=True)
    
    # make prediction
    prediction = knn.kneighbors(audio_feats_scaled)

    # create an index for similar songs
    similar_songs_index = prediction[1][0][:25].tolist()
    
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
    for i in range(25):
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

    # Remove the suggestion of the same song (all 0's)
    diff_df = diff_df[~(diff_df == 0).any(axis=1)]

    # Grab only the unique 10 songs
    diff_df = diff_df.drop_duplicates(subset=['sum'])[:10]

    diff_df = diff_df.reset_index(drop=True)


    return diff_df


worst_nites = spotify.iloc[77647].tolist()

test_audio_features = worst_nites

diff_df = knn_predictor(test_audio_features)

diff_json = diff_df.to_json(orient='records')

print(diff_json)