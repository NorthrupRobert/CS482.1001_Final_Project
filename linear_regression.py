import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Note: Linear Regression is currently using things that mix both Youtube & Spotify features. We just got to alter the irrelevant columns section in data cleaning.

class MyLinearRegression:
    def __init__(self, filepath):
        self.X = None
        self.y = None
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

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train(self):
        self.data_cleaning()
        self.read_csv()
        self.model_fit()

    def model_fit(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
    
    def model_predict(self):
        self.train()
        r_score = 0
        r_score = self.model.score(self.X_test, self.y_test)
        return r_score
    
    def predict_song(self, song_features):
        features_df = pd.DataFrame([song_features])

        feature_columns = self.data.drop(columns=['Stream']).columns
        features_df = features_df[feature_columns]

        scaled_features = self.scaler.transform(features_df)

        prediction = self.model.predict(scaled_features)
        return prediction[0]
    
def main():
    filepath = "./Spotify_Youtube.csv"
    regression = MyLinearRegression(filepath=filepath)
    regression.train()

    r_score = regression.model_predict()
    print(f"R-Score: {r_score:.4f}")

    new_song = {
        'Danceability': 0.7,
        'Energy': 0.8,
        'Key': 5,
        'Loudness': -5.3,
        'Mode': 1,
        'Speechiness': 0.05,
        'Acousticness': 0.1,
        'Instrumentalness': 0.0,
        'Liveness': 0.15,
        'Valence': 0.6,
        'Tempo': 120.0,
        'Duration_ms': 210000,
        'Views': 1000000,
        'Likes': 50000,
        'Comments': 2000
    }

    prediction = regression.predict_song(new_song)
    print(f"Predicted Streams for the new song: {prediction:.0f}")

if __name__ == "__main__":
    main()