import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import spotipy
from spotipy.oauth2 import SpotifyOAuth


#https://colab.research.google.com/drive/1llYC2qlo887tS7JSYDyCRXMNW7b80Jgk?usp=sharing#scrollTo=0DlXNWuZYUnP

class XGBoost:
    def __init__(self, filepath):
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.y_test = None
        self.X_test = None
        self.model = None
        self.data = pd.read_csv("./Spotify_Youtube.csv")

    def data_cleaning(self):
        irrelevant_columns = ['Track', 'Artist', 'Url_spotify', 'Uri', 'Url_youtube', 
                              'Title', 'Channel', 'Description', 'Licensed', 'official_video', 'Unnamed: 0']
        self.data = self.data.drop(columns=irrelevant_columns, errors='ignore')
        self.data = self.data.dropna()
        self.data = self.data.select_dtypes(include=[np.number])

    def read_csv(self):
        self.y = self.data['Stream']
        self.X = self.data.drop(columns=['Stream'])

        self.scaler = MinMaxScaler()
        self.X = self.scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)

    def train(self):
        self.data_cleaning()
        self.read_csv()
        params = {'objective': 'reg:squarederror', 'learning_rate': 0.05,'max_depth': 8,'subsample': 0.8, 'gamma': 1.0,}
        self.model = xgb.train(params, self.dtrain, num_boost_round=1000)

    def model_predict(self):
        self.predictions = self.model.predict(self.dtest)
        train_predictions = self.model.predict(self.dtrain)
        r_score = r2_score(self.y_test, self.predictions)
        train_r_score = r2_score(self.y_train, train_predictions)
        mse = mean_squared_error(self.y_test, self.predictions)
        mae = mean_absolute_error(self.y_test, self.predictions)
        return  train_r_score, r_score, mae, mse
    
    def plot(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.predictions)
        max_val = max(max(self.y_test), max(self.predictions))
        min_val = min(min(self.y_test), min(self.predictions))
        plt.plot([min_val, max_val], [min_val, max_val], color = 'red', label = 'Line of Best Fit')
        plt.xlabel('Actual Streams')
        plt.ylabel('Predicted Streams')
        plt.title('Actual vs Predicted Streams')

        plt.savefig('xgboost.png')
        plt.close()

    def predict_song(self, new_song):
        new_song_df = pd.DataFrame([new_song])
        
        scaled_song = self.scaler.transform(new_song_df)
        
        dnew_song = xgb.DMatrix(scaled_song)
        
        prediction = self.model.predict(dnew_song)
        
        return prediction

def main():
    filepath = "./Spotify_Youtube.csv"
    xgboost = XGBoost(filepath=filepath)
    
    xgboost.train()
    train_r_score, r_score, mse, mae = xgboost.model_predict()
    print(f"Training R2 Score: {train_r_score:}")
    print(f"Testing R2 Score: {r_score:}")
    print(f"Mean Squared Error (MSE): {mse:}")
    print(f"Mean Absolute Error (MAE): {mae:}")

    xgboost.plot()
    
    #track_uri = input("Enter the Spotify track URI: ")  # e.g., spotify:track:5KDP23HRuA9jETVYdZtA26
    #new_song = getUserSongFeatures(track_uri)
    #prediction = xgboost.predict_song(new_song)
    #print(f"Prediction: {int(prediction.item())}")

'''
class SongFeatures:
    def __init__(self):
        self.client_id = "234ca4a4bb3e4caabf7526822d85c7fb"
        self.client_secret = "8519ec11908b4d69baf547dc44d76155"
        self.redirect_uri = "http://localhost:8888/callback"
        self.sp = self.authenticate_spotify()

    def authenticate_spotify(self):
        sp_oauth = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope="user-library-read playlist-read-private streaming"
        )
        sp = spotipy.Spotify(auth_manager=sp_oauth)
        return sp

    def get_song_features(self, track_uri):
        try:
            token_info = self.sp.auth_manager.get_cached_token()
            if not token_info:
                print("No cached token, getting new token...")
                token_info = self.sp.auth_manager.get_access_token()
            
            if token_info:
                access_token = token_info['access_token']
                self.sp = spotipy.Spotify(auth=access_token)
                print("Access Token:", access_token)
            else:
                print("Unable to retrieve access token.")
                return None

            features = self.sp.audio_features(track_uri)
            if features and len(features) > 0:
                print("Features fetched successfully:", features[0])
                return features[0]
            else:
                print(f"No features found for track {track_uri}")
                return None

        except spotipy.exceptions.SpotifyException as e:
            print(f"Error fetching audio features: {e}")
            return None

def getUserSongFeatures(track_uri):
    song_features = SongFeatures()
    features = song_features.get_song_features(track_uri)

    if features is None:
        print(f"Error: No features returned for track {track_uri}")
        return None

    try:
        feature_data = {
            'Danceability': features['danceability'],
            'Energy': features['energy'],
            'Key': features['key'],
            'Loudness': features['loudness'],
            'Mode': features['mode'],
            'Speechiness': features['speechiness'],
            'Acousticness': features['acousticness'],
            'Instrumentalness': features['instrumentalness'],
            'Liveness': features['liveness'],
            'Valence': features['valence'],
            'Tempo': features['tempo']
        }
        return feature_data
    except KeyError as e:
        print(f"Missing expected feature: {e}")
        return None
'''
        
if __name__ == "__main__":
    main()