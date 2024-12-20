### A program that  will accept an audio file and transform to a linear spectrogram (trying to avoid the mel spectrogram s seems to be absed on bringing in line with human hearing range? seems unnecessary for this usecase)
### TODO: create a chart that generates whilst playing the file, to count gunshots.

import librosa
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

file_path = input("Insert File Path: ")
y, sr = librosa.load(file_path, sr=None)
# Apply Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Convert to amplitude spectrogram
S_db = librosa.amplitude_to_db(abs(D))

# Plot the spectrogram
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear Spectrogram')
plt.show()