import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

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

        self.scaler = StandardScaler()
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
        predictions = self.model.predict(self.dtest)
        r_score = r2_score(self.y_test, predictions)
        return  r_score
    
def main():
    filepath = "./Spotify_Youtube.csv"
    xgboost = XGBoost(filepath=filepath)
    
    xgboost.train()
    r_score = xgboost.model_predict()
    print(f"R2 Score: {r_score:.4f}")

if __name__ == "__main__":
    main()