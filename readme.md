Testing audio streaming with a small convolutional neural network to try and classify streamed sounds.

Current target: Using a small Convolutional Neural Network trained with Tensorflow, and packaged for deployment to an SBC using tensorflow-lite.

Current approach, 8732 sounds downloaded from kaggle, with approx. 13% of the dataset being gunshots. Applied preprocessing and model in line with the forked repository from this github -> https://github.com/mariamkhmahran/gunshot-detection-system/blob/main/README.md
From Mariam's documentation, it seems like a simple fast fourier transformation model is not accurate enough to be valuable. Therefore it will probably be necessary to  extract more features from  the  dataset. As per the workflow Mariam posted. Unfortunately I ran into many issues carrying out the preprocessing, one being that I could not load the entire dataset  to extract features, for expediency I ran folders about 4000 sounds for the training data, with 1000 each for validation and testing. Some of the pre processing steps also did not work for me. Resulting in a worse training data set. I may need to work on the feature extraction from the ground up. 

For training the model, I ran the SVM model on the processed data, producing a model with ~20% accuracy, which will be unsuitable for our use case. Additionally running a CNN model produced even worse results. This may be as it required me to perform additional processing steps to get the data in the correct format for training, and I do not fully understand the implications of the changes made to the data. 

Considering the above, I am aiming to create a multi-feature extraction from the same data set, but from the ground up, then run another small CNN based on the feature set. Currrently I think a complete ground up rework and refresh of extraced features will be the only way to get a more reliable model.
