# BWSML Project: Predicting Song Popularities on Spotify

This project was developed for the [Bern Winter School on Machine Learning](https://www.math.unibe.ch/continuing_education/cas_applied_data_science/winter_school_on_machine_learning_2020/index_eng.html), which is organized by the Mathematical Institute of the University of Bern. The project builds on a dataset from [Kaggle](https://www.kaggle.com/edalrami/19000-spotify-songs) that contains descriptive information of almost 19'000 songs from Spotify. I have used the package ```sotifyR``` to collect additional data through the [Spotify API](https://developer.spotify.com/documentation/web-api/) to construct a dataset that also features information on release dates of the songs. I have then used Tensorflow and Keras to train different neural networks that predict, whether a given song is a popular song on Spotify.

**Note** that this is a fun project with no deeper meaning. It's purpose was for me to try out different neural networks as well as to get more familiar with Tensorflow and Keras. 

## Thoughts and conclusions

The accuracies of these models were around 70%, which indicates that the networks are somewhat able to differentiate between popular and non-popular songs. Better performance could be expected by applying the approach to specific music genres or decades. This would require building a different dataset, which was outside the scope of this project. Ultimately, I would expect to have a much better performance when changing the features. Instead of relying on song characteristics, one could use the actual signal of songs. This would capture things like the actual speed, melody etc. of a song, which I would expect to be the most important predictors for song popularities. Such data could be used for a CNNs or LSTMs.

## Material
The repository contains the following material:

- A presentation of the results (see `Slides_Spotify_Songs.pdf`)
- All the datasets used for this project (`data.zip`)
- The R script ```spotify_scrap.R``` that I have used to access the spotify API and to collect additional data.
- The Python script ```sptfy_nn_project.py``` that was used for data processing, training and evaluating the models.

