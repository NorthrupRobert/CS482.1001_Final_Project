# -*- coding: utf-8 -*-
"""CS482.1001 FP NN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1A3Rzqelia2bBcd11E2Z3_EDbB0CkCSzK
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

class MySuperNeuralNetwork:
    def __init__(self, filepath):
        self.X = None
        self.y = None
        self.y_test = None
        self.X_test = None
        self.model = None
        self.scaler = None
        self.history = None
        self.data = pd.read_csv(filepath)

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
        self.y = self.y.values.reshape(-1, 1)
        self.y = self.scaler.fit_transform(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train(self):
        self.data_cleaning()
        self.read_csv()
        self.model_fit()

    def model_fit(self):
        self.model = Sequential([
            Input(shape=(self.X_train.shape[1],)),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(1, activation='linear')
        ])

        optimizer = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        self.history = self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32,
                                      validation_split=0.2, callbacks=[lr_scheduler, early_stopping])

    def model_predict(self):
        predictions = self.model.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_inverse = self.scaler.inverse_transform(self.y_test)
        mse = mean_squared_error(y_test_inverse, predictions)
        r2 = r2_score(y_test_inverse, predictions)
        mae = mean_absolute_error(y_test_inverse, predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')
        print(f'Mean Absolute Error: {mae}')
        self.plot_metrics()

    def plot_metrics(self):
        plt.figure(figsize=(14, 10))

        # Plot training & validation loss values
        plt.subplot(2, 1, 1)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Test Loss')
        plt.title('Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        # Plot training & validation accuracy values
        plt.subplot(2, 1, 2)
        plt.plot(self.history.history.get('accuracy', []), label='Train Accuracy')
        plt.plot(self.history.history.get('val_accuracy', []), label='Test Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

    def predict_song(self, song_features):
        # Map the input to a DataFrame with the original data columns
        irrelevant_columns = ['Track', 'Artist', 'Url_spotify', 'Uri', 'Url_youtube',
                              'Title', 'Channel', 'Description', 'Licensed',
                              'official_video', 'Unnamed: 0']

        # Create a DataFrame to process the input like the training data
        input_data = pd.DataFrame([song_features], columns=self.data.columns, dtype=object)

        # Remove irrelevant columns
        input_data = input_data.drop(columns=irrelevant_columns, errors='ignore')

        # Ensure only numerical columns remain
        input_data = input_data.select_dtypes(include=[np.number])

        # Check if the input has the correct number of features
        if input_data.shape[1] != self.X.shape[1]:
            raise ValueError(f"Expected {self.X.shape[1]} numerical features, but got {input_data.shape[1]}.")

        # Scale the features
        song_features_scaled = self.scaler.transform(input_data)

        # Make predictions
        predicted_streams = self.model.predict(song_features_scaled)
        predicted_streams = self.scaler.inverse_transform(predicted_streams)
        return predicted_streams[0]

    def calculate_training_r2(self):
        # Predict on the training set
        predictions_train = self.model.predict(self.X_train)
        predictions_train = self.scaler.inverse_transform(predictions_train)
        y_train_inverse = self.scaler.inverse_transform(self.y_train)

        # Calculate R-squared
        r2_train = r2_score(y_train_inverse, predictions_train)
        print(f'Training R-squared: {r2_train}')
        return r2_train

# USAGE FOR UPDATED NEURAL NETWORK
filepath = './Spotify_Youtube.csv'
nn_model_s = MySuperNeuralNetwork(filepath)
nn_model_s.train()
nn_model_s.model_predict()
nn_model_s.calculate_training_r2()