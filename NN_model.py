import pandas as pd
import numpy as np
import json
from joblib import dump
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load in the data
spotify = pd.read_csv('https://raw.githubusercontent.com/BW-pilot/MachineLearning/master/CSVs/spotify_final.csv')
print(spotify.head())


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def predict(model, input_vector):
    return model.predict(input_vector).argsort()

def build_model(weights):
    model = Sequential([
        # Dot product between feature vector and reference vectors
        Dense(input_shape=(weights.shape[1],),
              units=weights.shape[0],
              activation='linear',
              name='dense_1',
              use_bias=False)
    ])
    model.set_weights([weights.T])
    return model

# ,acousticness,artists,danceability,duration_ms,energy,explicit,id,instrumentalness,key,liveness,loudness,mode,name,popularity,release_date,speechiness,tempo,valence,year
def get_results(input_vector, features, best_match=True, amount=5):
    """
    get_results(input_vector, features, best_match=True, amount=5)
    input_vector: audio features of the song to suggest similar songs to,
    plus track_id
    features: full database to suggest songs from
    best_match=True: True if you want most similar songs, False if least
    similar
    amount=5: amount of results to return.
    returns a list (might be a numpy array?) of indices from the original
    database
    """

    # column names that will be used in training / ID is there for later use
    col_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                 'liveness','loudness', 'tempo', 'valence', 'id']

    # create input vector for the model
    input_vector_df = pd.DataFrame([input_vector], columns=col_names)
    
    # save IDs to an array of track_ids
    track_id = input_vector_df['id'].values[0]

    # save ALL track IDs to a series for later use
    ids = features['id']

    # drop 'id' cols from input vector and all_songs_df
    input_vec = input_vector_df.drop(columns=['id'])
    feats = features.drop(columns=['id'])

    # norm_vector = normalize(input_vec.values)
    norm_vector = normalize(input_vec)
    norm_features = normalize(feats)

    # instantiate the model and make predictions
    model = build_model(norm_features)
    prediction = np.array(predict(model, norm_vector).argsort())
    prediction = prediction.reshape(prediction.shape[1])

    # Add back 'ID' onto the end of the output
    feats['id'] = ids

    # Make sure best suggestion isn't the same song as the input song
    if best_match:
        if track_id in ids[prediction[-amount:]]:
            return feats.loc[prediction[-amount-1:-1]]
        return feats.loc[prediction[-amount:]]
    return feats.loc[prediction[:amount]]

worst_nites = df.iloc[77647].tolist()
test_audio_features = worst_nites
# test_audio_features = [0.5,	0.7, 0.7, 0.0, 3, 0.1, -3, 0.03, 130, 0.9, '6oXghnUUe9u2iIZPNfCxjl']   

# get results from first song in database
results_1 = get_results(df.iloc[0], df, amount=5)
print('-------------------------')
print(results_1)

# get results from test_audio features, which is actually the song Worst Nites by Foster the People
results_2 = get_results(test_audio_features, df, amount=10)
print('-------------------------')
print(results_2)