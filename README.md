# BWSML Project: Predicting Song Popularities on Spotify

This project was developed for the [Bern Winter School on Machine Learning](https://www.math.unibe.ch/continuing_education/cas_applied_data_science/winter_school_on_machine_learning_2020/index_eng.html), which is organized by the Mathematical Institute of the University of Bern. The project builds on a dataset from [Kaggle](https://www.kaggle.com/edalrami/19000-spotify-songs) that contains descriptive information of almost 19'000 songs from *Spotify*. I have used the package ```sotifyR``` to collect additional data through the [Spotify API](https://developer.spotify.com/documentation/web-api/) to construct a dataset that also features information on release dates of the songs. With this, I have then used tensorflow to train several neural networks to predict, whether a specific song is a popular song on *Spotify*. 

Note that this is a fun project with no deeper meaning. It's purpose was for me to try out different neural networks as well as to get more familiar with Tensorflow and Keras.

## Material
The repository contains the following material:

- A presentation of the results (see `Slides_Spotify_Songs.pdf`)
- All the datasets used for this project (`data.zip`)
- The R script ```spotify_scrap.R``` that I have used to access the spotify API and to collect additional data.
- The Python script ```sptfy_nn_project.py``` that was used for data processing, training and evaluating the models.

