import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import tensorflow as tf

# %% load the data

path = "xxxxxxxxxxxxxxxxxxxxxx" # get the path to the datasets

song_dat = pd.read_csv(path+"/song_data.csv") # contains song characteristics and popularity scores
song_info =pd.read_csv(path+"/song_info.csv") # contains artist and album information of songs
album_dates = pd.read_csv(path+"/album_dates.csv", index_col = 0) # contains album release date information
tf.random.set_seed(1234) # set seed for reproducibility

# %% DATA CLEANING
print("Number of songs: ", len(song_dat))
song_dat.head()

# Some songs are present multiple times in the raw data. 
# This is because they are from different albums by the same artist. 
# Therefore, drop all songs that have the same 'song_name', 'song_duration' (in milli-seconds) and 'acousticness'.
song_dat.drop_duplicates(subset = ["song_name", "song_duration_ms", "acousticness"], inplace = True)
print("After dropping duplicated songs, there are", len(song_dat), "unique songs left in the sample")

# group song popularity in 4 bins
n_bins = 4
song_dat["song_pop_class"] = pd.cut(song_dat["song_popularity"], bins = n_bins, labels = False)
song_dat["top_song"] = np.where(song_dat["song_pop_class"] == n_bins -1, 1, 0)
song_dat["top_song"].value_counts()
song_dat["song_pop_class"].value_counts()

# %% Correlation Plot

# illustrate the data
df = song_dat[3:-2]
plt.matshow(df.corr())
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
plt.show()

#%% Prepare data for training

# convert everything to floats
pred_dat = song_dat
pred_dat.song_duration_ms = pred_dat.song_duration_ms.astype(float)
pred_dat.audio_mode = pred_dat.audio_mode.astype(float)
pred_dat.time_signature = pred_dat.time_signature.astype(float)
pred_dat.dropna(inplace= True)

# normalize features
x = pred_dat.iloc[:, 2:-2]
x = (x - np.min(x))/(np.max(x)-np.min(x)).values
pred_dat.iloc[:, 2:-2] = x
pred_dat = pred_dat.reset_index().drop(["index"], axis =1) # evtl. wieder weg

# split the data to train and test samples:
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(pred_dat.iloc[:, 2:], test_size = 0.2)

y_train = x_train["top_song"].values
x_train.drop(["top_song", "song_pop_class"], axis = 1, inplace = True)
x_train = x_train.values

y_test = x_test["top_song"].values
test_ids = x_test.index # keep the indexes for validation!!
x_test.drop(["top_song", "song_pop_class"], axis = 1, inplace = True)
x_test = x_test.values

print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("x_test: ",x_test.shape)
print("y_test: ",y_test.shape)

#%% train and evaluate Model I

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(30, activation='relu'),  #  30 neurons, ReLU activation
  tf.keras.layers.Dense(10, activation='relu'),   # 10 neurons, ReLU activation
  tf.keras.layers.Dense(2, activation='softmax')  # 2 outputs with softmax activation: probability of being a popular song
])

model.compile(optimizer='adam',                        # Adam optimizer
              loss='sparse_categorical_crossentropy',  # objective function - cross-entropy, class-ID labels
              metrics=['accuracy'])                    # additional performance metrices

hist = model.fit(x = x_train, y = y_train,
                 epochs = 25, batch_size = 64, 
                 validation_data=[x_test, y_test])

fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['accuracy'])
axs[1].plot(hist.epoch, hist.history['val_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()

model.evaluate(x_test,  y_test, verbose=2)
#%% evaluate accuracies

# check the prediction for the top rated song in the test set
songs = pred_dat.iloc[test_ids, :]
song_id = songs.sort_values(by = "song_popularity", ascending = False)[0:1].index[0]
test_songs = songs.reset_index()
song_id = test_songs[test_songs["index"] == song_id].index[0]
y_pred = model(x_test[song_id:song_id+1])
print('song: ', str(test_songs[song_id:song_id+1]["song_name"]))#, 'artist: ', list(test_songs[song_id[0]:song_id[0]+1]["artist_name"]))
print('true lablel: ', y_test[song_id], 'predicted: ', np.argmax(y_pred[0]))

# check the prediction for the top-ten songs
song_id = list(songs.sort_values(by = "song_popularity", ascending = False)[0:10].index)
test_songs = songs.reset_index()
song_id = test_songs[test_songs["index"].isin(song_id) == True].index
y_pred = model(x_test[song_id])
pred = list(map(lambda x: np.argmax(x), y_pred))
actual = list(y_test[song_id])
list(map(lambda x, y: x==y, pred, actual))

# check for all top songs (precision)
song_id = songs.sort_values(by = "song_popularity", ascending = False)
song_id = song_id[song_id.top_song == 1]
song_id = song_id.index
test_songs = songs.reset_index()
song_id = test_songs[test_songs["index"].isin(song_id) == True].index
y_pred = model(x_test[song_id])
pred = list(map(lambda x: np.argmax(x), y_pred))
actual = list(y_test[song_id])
acc = list(map(lambda x, y: x==y, pred, actual))
acc = acc.count(True) / len(acc)
print("Accuracy among top songs in the test set is: ", round(100 * acc, 2), "%")
print("Share of top songs in the training sample is ", round(100 * len(y_train[y_train == 1]) / len(y_train),2), "%")

#%% Rebalance the sample
pred_dat = song_dat
pred_dat.song_duration_ms = pred_dat.song_duration_ms.astype(float)
pred_dat.audio_mode = pred_dat.audio_mode.astype(float)
pred_dat.time_signature = pred_dat.time_signature.astype(float)
pred_dat.dropna(inplace= True)

# normalize all features ---------------
x = pred_dat.iloc[:, 2:-2]
x = (x - np.min(x))/(np.max(x)-np.min(x)).values
pred_dat.iloc[:, 2:-2] = x
pred_dat = pred_dat.reset_index().drop(["index"], axis =1)

# balance the sample and define weights for the loss ------------------
# specify the desired ratio between non-top and top-songs in the sample
ratio = 4
# ratio = round(len(pred_dat) / len(pred_dat[pred_dat.top_song == 1])) # for keeping the original distribution

# define the weights for the groups in the loss-function
cl_weight = {0:1, 1:(ratio-0.75)} # this turned out to be a good weighting scheme

# sample the data accordingly
pred_dat0 = pred_dat[pred_dat.top_song == 1]
pred_dat1 = pred_dat[pred_dat.top_song == 0].sample(frac = 1).reset_index(drop = True)
n_nonTopSong = len(pred_dat0) * ratio
pred_dat1 = pred_dat1.iloc[0:n_nonTopSong, :]
pred_dat = pd.concat([pred_dat0, pred_dat1])
pred_dat = pred_dat.reset_index(drop = True)

# split to train and test samples: ------------
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(pred_dat.iloc[:, 2:], test_size = 0.2)

# training sample ----------------
y_train = x_train["top_song"].values
x_train.drop(["top_song", "song_pop_class"], axis = 1, inplace = True)
x_train = x_train.values

# test sample -----------------
y_test = x_test["top_song"].values
test_ids = x_test.index # keep the indexes for validation!!
x_test.drop(["top_song", "song_pop_class"], axis = 1, inplace = True)
x_test = x_test.values

print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("x_test: ",x_test.shape)
print("y_test: ",y_test.shape)

#%% re-train and evaluate the model (Model II)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(30, activation='relu'),  #  39 neurons, ReLU activation
  tf.keras.layers.Dense(10, activation='relu'),   # 13 neurons, ReLU activation
  tf.keras.layers.Dense(2, activation='softmax')  # 2 outputs with softmax activation: probability of being a popular song
])

model.compile(optimizer='adam',                        # Adam optimizer
              loss='sparse_categorical_crossentropy',  # objective function - cross-entropy, class-ID labels
              metrics=['accuracy'])

hist = model.fit(x = x_train, y = y_train,
                 epochs = 25, batch_size = 32, 
                 validation_data=[x_test, y_test], class_weight = cl_weight)

fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['accuracy'])
axs[1].plot(hist.epoch, hist.history['val_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()

model.evaluate(x_test,  y_test, verbose=2)[1]

#%% check accuracies of Model II

def top_song():
    songs = pred_dat.iloc[test_ids, :]
    song_id = songs.sort_values(by = "song_popularity", ascending = False)
    song_id = song_id[song_id.top_song == 1]
    song_id = song_id.index
    test_songs = songs.reset_index()
    song_id = test_songs[test_songs["index"].isin(song_id) == True].index
    y_pred = model(x_test[song_id])
    pred = list(map(lambda x: np.argmax(x), y_pred))
    actual = list(y_test[song_id])
    acc = list(map(lambda x, y: x==y, pred, actual))
    acc = acc.count(True) / len(acc)
    return acc
acc1 = top_song()
print("Accuracy among top songs in the test set is: ", round(100 * acc1, 2), "%")
print("Share of top songs in the training sample is ", round(100 * len(y_train[y_train == 1]) / len(y_train),2), "%")

def false_pos():
    songs = pred_dat.iloc[test_ids, :]
    song_id = songs.sort_values(by = "song_popularity", ascending = False)
    song_id = song_id[song_id.top_song == 0]
    song_id = song_id.index
    test_songs = songs.reset_index()
    song_id = test_songs[test_songs["index"].isin(song_id) == True].index
    y_pred = model(x_test[song_id])
    pred = list(map(lambda x: np.argmax(x), y_pred))
    actual = list(y_test[song_id])
    acc = list(map(lambda x, y: x==y, pred, actual))
    acc = acc.count(False) / len(acc)
    return acc
acc2 = false_pos()
print("False Positives in the test set: ", round(100 * acc2, 2), "%")
print("Share of non-top songs in the training sample is ", round(100 * len(y_train[y_train == 0]) / len(y_train),2), "%")

#%% Illustrations: get new spotify songs and make a prediction
illus_songs = pd.read_csv(path+"/illustration_songs.csv")
illus_songs = illus_songs.iloc[:, 2:15]
illus_songs.song_duration_ms = illus_songs.song_duration_ms.astype(float)
illus_songs.audio_mode = illus_songs.audio_mode.astype(float)
illus_songs.time_signature = illus_songs.time_signature.astype(float)

# standardize
illus_songs = (illus_songs - np.min(illus_songs))/(np.max(illus_songs)-np.min(illus_songs)).values
illus_songs["time_signature"] = 0.8
illus_songs = illus_songs.values
y_pred = model(illus_songs)
y_pred

#%% FEATURE IMPORTANCE: estimate by using permutation importance

# this block of code evaluates feature importance by retraining models

# ------------------ NOT RUN FROM HERE !!! ------------------------------

# get the performance metrics from the baseline model (Model II)
overall_acc = round(model.evaluate(x_test,  y_test, verbose=2)[1],3)
top_song_acc = round(top_song(), 3)
fp_nonTop = round(false_pos(), 3)
model_acc = {"overall_acc": overall_acc, "top_song_acc": top_song_acc, "fp_nonTop": fp_nonTop}
model_acc = pd.DataFrame(model_acc, index = ["full model"])    

# for each iteration, shuffel one feature in x_train
# and evaluate the model trained with the shuffeld x_train data
# store the results in the dataframe 'model_acc'
n_vars = len(x_train[0])
feature_names = pred_dat.columns[3:-2]

for j in range(n_vars-1):
    
    # prepare the data and shuffle
    x_train2 = x_train
    xs = [x_train2[i][0] for i in range(len(x_train2)-1)]
    random.shuffle(xs)
    for i in range(len(x_train2)-1):
        x_train2[i][j] = xs[i]
    
    # train the network
    hist = model.fit(x = x_train2, y = y_train,
                 epochs = 25, batch_size = 32, 
                 validation_data=[x_test, y_test], class_weight = cl_weight)
    
    # evaluate the network
    overall_acc = round(model.evaluate(x_test,  y_test, verbose=2)[1],3)
    top_song_acc = round(top_song(), 3)
    fp_nonTop = round(false_pos(), 3)
    accs = {"overall_acc": overall_acc, "top_song_acc": top_song_acc, "fp_nonTop": fp_nonTop}
    accs = pd.DataFrame(accs, index = [feature_names[j]])    
    
    # store the results
    model_acc = pd.concat([model_acc, accs], axis = 0)

# all results
model_acc

# calculate performance changes relative to the baseline model 
feature_imp = pd.concat([model_acc["overall_acc"]/model_acc.loc["full model","overall_acc"],
           model_acc["top_song_acc"]/model_acc.loc["full model","top_song_acc"],
           model_acc["fp_nonTop"]/model_acc.loc["full model","fp_nonTop"]], axis = 1)

# ------------------ NOT RUN UNTILL HERE !!! ------------------------------

#%% PLOT FEATURE IMPORTANCE 
feature_imp = pd.read_csv(path+"/feature_imp.csv")
feature_imp = feature_imp.set_index([feature_imp.columns[0]])
feature_imp = pd.concat([feature_imp["overall_acc"]/feature_imp.loc["full model","overall_acc"],
           feature_imp["top_song_acc"]/feature_imp.loc["full model","top_song_acc"],
           feature_imp["fp_nonTop"]/feature_imp.loc["full model","fp_nonTop"]], axis = 1)
feature_imp

# plot performance changes for overall accuracy
plt.bar(range(feature_imp[1:].shape[0]), feature_imp[feature_imp.columns[0]][1:])
plt.xticks(range(feature_imp[1:].shape[0]), list(feature_imp.index[1:]), rotation = 45)
plt.title("Overall accuracy")

# plot performance changes for top song accuracy
plt.bar(range(feature_imp[1:].shape[0]), feature_imp[feature_imp.columns[1]][1:])
plt.xticks(range(feature_imp[1:].shape[0]), list(feature_imp.index[1:]), rotation = 45)
plt.title("Accuracy among top songs")

# plot performance changes for false positives of non-top songs
plt.bar(range(feature_imp[1:].shape[0]), feature_imp[feature_imp.columns[2]][1:])
plt.xticks(range(feature_imp[1:].shape[0]), list(feature_imp.index[1:]), rotation = 45)
plt.title("False positives on non-popular songs")


#%% RECENT SONGS: Subset to songs released after 2010 (Model III)

# Reload original dataset of all songs and combine them with album and artist information ----------------
song_dat2 = pd.read_csv(path+"/song_data.csv")
# combine song characteristics and song metadata
song_dat2 = pd.concat([song_dat2, song_info.iloc[:,1:]], axis = 1)
# rearrage columns:
col_names = song_dat2.columns.tolist()[0:1]+song_dat2.columns.tolist()[-3:]+song_dat2.columns.tolist()[1:-3]
song_dat2 = song_dat2[col_names]

# summarize
print("number of unique songs: ", len(song_dat2))
song_dat2.sort_values(by = "song_popularity", ascending = False).head(10)
song_dat2 = song_dat2.reset_index().drop(["index"], axis = 1)
song_dat2.head()

# Load data for album release dates from spotify API
album_dates = album_dates.iloc[:, 0:4]
album_dates = album_dates.drop_duplicates(subset = ["artist_name", "album_names"])

# add album release dates to dataset
song_dat2 = pd.merge(song_dat2, album_dates, how = "left", on = ["artist_name", "album_names"])
song_dat2 = song_dat2[np.isnan(song_dat2.album_release_year) == False]
song_dat2 = song_dat2[song_dat2.album_release_year >= 2010]
song_dat2 = song_dat2.drop(["album_release_year", "album_release_date"], axis = 1)

# group song popularity in bins
n_bins = 4
song_dat2["song_pop_class"] = pd.cut(song_dat2["song_popularity"], bins = n_bins, labels = False)
song_dat2["top_song"] = np.where(song_dat2["song_pop_class"] == n_bins -1, 1, 0)
song_dat2["song_pop_class"].value_counts()

# prepare the data for training ------------------------
pred_dat = song_dat2
pred_dat.song_duration_ms = pred_dat.song_duration_ms.astype(float)
pred_dat.audio_mode = pred_dat.audio_mode.astype(float)
pred_dat.time_signature = pred_dat.time_signature.astype(float)
pred_dat.dropna(inplace= True)

# normalize all features ---------------
x = pred_dat.iloc[:, 5:-2]
x = (x - np.min(x))/(np.max(x)-np.min(x)).values
pred_dat.iloc[:, 5:-2] = x
pred_dat = pred_dat.reset_index().drop(["index"], axis =1)

# balance the sample and define weights for the loss ------------------
ratio = 4 # specify the desired ratio between non-top and top-songs in the sample
# ratio = round(len(pred_dat) / len(pred_dat[pred_dat.top_song == 1])) # keep original distribution

# define the according weights for the loss-function
cl_weight = {0:1, 1:(ratio-0.5)}

# sample the data accordingly
pred_dat0 = pred_dat[pred_dat.top_song == 1]
pred_dat1 = pred_dat[pred_dat.top_song == 0].sample(frac = 1).reset_index(drop = True)
n_nonTopSong = len(pred_dat0) * ratio
pred_dat1 = pred_dat1.iloc[0:n_nonTopSong, :]
pred_dat = pd.concat([pred_dat0, pred_dat1])
pred_dat = pred_dat.reset_index(drop = True)

# split to train and test samples: ------------
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(pred_dat.iloc[:, 5:], test_size = 0.2)

# training sample ----------------
y_train = x_train["top_song"].values
x_train.drop(["top_song", "song_pop_class"], axis = 1, inplace = True)
x_train = x_train.values

# test sample -----------------
y_test = x_test["top_song"].values
test_ids = x_test.index # keep the indexes for validation!!
x_test.drop(["top_song", "song_pop_class"], axis = 1, inplace = True)
x_test = x_test.values

print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("x_test: ",x_test.shape)
print("y_test: ",y_test.shape)

#%% train and evaluate Model III
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(30, activation='relu'),  #  39 neurons, ReLU activation
  tf.keras.layers.Dense(10, activation='relu'),   # 13 neurons, ReLU activation
  tf.keras.layers.Dense(2, activation='softmax')  # 2 outputs with softmax activation: probability of being a popular song
])

model.compile(optimizer='adam',                        # Adam optimizer
              loss='sparse_categorical_crossentropy',  # objective function - cross-entropy, class-ID labels
              metrics=['accuracy'])

hist = model.fit(x = x_train, y = y_train,
                 epochs = 25, batch_size = 32, 
                 validation_data=[x_test, y_test], class_weight = cl_weight)

fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['accuracy'])
axs[1].plot(hist.epoch, hist.history['val_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()

model.evaluate(x_test,  y_test, verbose=2)[1]

#%% check accuracies of Model III
def top_song():
    songs = pred_dat.iloc[test_ids, :]
    song_id = songs.sort_values(by = "song_popularity", ascending = False)
    song_id = song_id[song_id.top_song == 1]
    song_id = song_id.index
    test_songs = songs.reset_index()
    song_id = test_songs[test_songs["index"].isin(song_id) == True].index
    y_pred = model(x_test[song_id])
    pred = list(map(lambda x: np.argmax(x), y_pred))
    actual = list(y_test[song_id])
    acc = list(map(lambda x, y: x==y, pred, actual))
    acc = acc.count(True) / len(acc)
    return acc
acc1 = top_song()
print("Accuracy among top songs in the test set is: ", round(100 * acc1, 2), "%")
print("Share of top songs in the training sample is ", round(100 * len(y_train[y_train == 1]) / len(y_train),2), "%")

def false_pos():
    songs = pred_dat.iloc[test_ids, :]
    song_id = songs.sort_values(by = "song_popularity", ascending = False)
    song_id = song_id[song_id.top_song == 0]
    song_id = song_id.index
    test_songs = songs.reset_index()
    song_id = test_songs[test_songs["index"].isin(song_id) == True].index
    y_pred = model(x_test[song_id])
    pred = list(map(lambda x: np.argmax(x), y_pred))
    actual = list(y_test[song_id])
    acc = list(map(lambda x, y: x==y, pred, actual))
    acc = acc.count(False) / len(acc)
    return acc
acc2 = false_pos()
print("False Positives in the test set: ", round(100 * acc2, 2), "%")
print("Share of non-top songs in the training sample is ", round(100 * len(y_train[y_train == 0]) / len(y_train),2), "%")


#%%  EXTENSIONS: include popularity of previous albums of the same artist
    
# load the data -----------
song_dat3 = pd.read_csv(path+"/song_data.csv") # contains song characteristics and popularity scores
song_dat3 = pd.concat([song_dat3, song_info.iloc[:,1:]], axis = 1)
col_names = song_dat3.columns.tolist()[0:1]+song_dat3.columns.tolist()[-3:]+song_dat3.columns.tolist()[1:-3]
song_dat3 = song_dat3[col_names]
song_dat3 = song_dat3.reset_index().drop(["index"], axis = 1)
song_dat3 = pd.merge(song_dat3, album_dates, how = "left", on = ["artist_name", "album_names"])
song_dat3 = song_dat3[np.isnan(song_dat3.album_release_year) == False]

# extract artists that have at least 3 different albums in the dataset:
tmp = song_dat3.groupby(["artist_name"])["album_names"].nunique()
tmp = tmp[tmp >= 3]
song_dat3 = song_dat3[song_dat3.artist_name.isin(list(tmp.index))]

# calculate average album characteristics
song_dat3 = song_dat3.drop(["song_name", "playlist", "album_release_date"], axis = 1)
song_dat3 = song_dat3.groupby(["artist_name", "album_names"])
song_dat3 = song_dat3.mean()

# find the most recent 3 albums per artist
# and subset the data to these observations
tmp = song_dat3["album_release_year"].groupby(level = 0)
tmp = tmp.apply(lambda x: x.sort_values(ascending = False)[0:3])
tmp = tmp.reset_index(level = [1, 2], drop = False).reset_index(drop = True)
tmp = tmp.drop(["album_release_year"], axis = 1)
song_dat3 = song_dat3.reset_index()
song_dat3 = pd.merge(tmp, song_dat3, how = "left", on = ["artist_name", "album_names"])

print("number of artists: ", round(len(song_dat3)/3))
song_dat3.head(15)

# label the albums
n_bins = 4
song_dat3["song_pop_class"] = pd.cut(song_dat3["song_popularity"], bins = n_bins, labels = False)
song_dat3["top_song"] = np.where(song_dat3["song_pop_class"] == n_bins-1, 1, 0)
song_dat3["song_pop_class"].value_counts()

# normalize all features ---------------
pred_dat = song_dat3
pred_dat.song_duration_ms = pred_dat.song_duration_ms.astype(float)
pred_dat.audio_mode = pred_dat.audio_mode.astype(float)
pred_dat.time_signature = pred_dat.time_signature.astype(float)
pred_dat.dropna(inplace= True)

x = pred_dat.iloc[:, 3:-3]
x = (x - np.min(x))/(np.max(x)-np.min(x)).values
pred_dat.iloc[:, 3:-3] = x
pred_dat = pred_dat.reset_index().drop(["index"], axis =1)
pred_dat

#%% PAST POPULARITY (Model IV)

# create a sequence with the album values for every artist
# initialize the dataset:
n_vars = pred_dat.iloc[:, 3:-3].shape[1]
n_features = pred_dat.iloc[:, 3:-3].shape[1]
c_names = [('var%d(t)' % (j+1)) for j in range(n_vars)]
for i in range(1,3):
  c_names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
artists = list(np.unique(pred_dat["artist_name"]))
seq_dat = pd.DataFrame(columns = c_names)

# loop over all artists
for i in range(0, len(artists)):
  
  # subset for artist
  tmp = pred_dat[pred_dat.artist_name == artists[i]]

  # add lagged album values for every observation
  covars = tmp.iloc[:, 3:-3]
  covars = np.array(pd.concat([covars.iloc[0, ], covars.iloc[1, ], covars.iloc[2, ]], axis = 0))
  covars = pd.DataFrame(covars, index = c_names)
  covars = covars.transpose()

  # get outcomes:
  outcomes = pd.DataFrame(np.array(tmp["top_song"]), index = ["top_album(t)", "top_album(t-1)", "top_album(t-2)"]).transpose()
  
  # combine:
  tmp = pd.concat([outcomes, covars], axis = 1)

  # add to seq_dat
  seq_dat = pd.concat([seq_dat, tmp], axis = 0)

# set indexes
seq_dat = seq_dat.set_index([artists])

print("Time series containing information of the past 2 albums")
print("Number of artists: ", seq_dat.shape[0])
seq_dat.head(10)

# prepare for training: -------------------------
# split to train and test samples:
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(seq_dat, test_size = 0.2)

# training sample:
y_train = x_train["top_album(t)"].values
x_train.drop(["top_album(t)"], axis = 1, inplace = True)
x_train = x_train.values

# test sample:
y_test = x_test["top_album(t)"].values
test_ids = x_test.index # keep the indexes for validation!!
x_test.drop(["top_album(t)"], axis = 1, inplace = True)
x_test = x_test.values

print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("x_test: ",x_test.shape)
print("y_test: ",y_test.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(30, activation='relu'),  #  30 neurons, ReLU activation
  tf.keras.layers.Dense(10, activation='relu'),  #  10 neurons, ReLU activation
  tf.keras.layers.Dense(2, activation='softmax')  # 2 outputs with softmax activation: probability of being a popular song
])

ratio=len(seq_dat) // seq_dat["top_album(t)"].value_counts()[1]
cl_weight = {0:1, 1:(ratio)}

model.compile(optimizer='adam',                        # Adam optimizer
              loss='sparse_categorical_crossentropy',  # objective function - cross-entropy, class-ID labels
              metrics=['accuracy'])                    # additional performance metrices

hist = model.fit(x = x_train, y = y_train,
                 epochs = 25, batch_size = 8, 
                 validation_data=[x_test, y_test],  class_weight = cl_weight)

fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['accuracy'])
axs[1].plot(hist.epoch, hist.history['val_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()
model.evaluate(x_test,  y_test, verbose=2)

#%% check accuracies of Model IV
songs = seq_dat.loc[test_ids, :]
song_id = songs["top_album(t)"] == 1
song_id = song_id[song_id == True]
song_id = song_id.index
test_songs = songs.reset_index()
song_id = test_songs[test_songs["index"].isin(song_id) == True].index
y_pred = model(x_test[song_id])
pred = list(map(lambda x: np.argmax(x), y_pred))
actual = list(y_test[song_id])
acc = list(map(lambda x, y: x==y, pred, actual))
acc = acc.count(True) / len(acc)
print("Accuracy among top songs in the test set is: ", round(100 * acc, 2), "%")
print("Share of top songs in the training sample is ", round(100 * len(y_train[y_train == 1]) / len(y_train),2), "%")

song_id = songs["top_album(t)"] == 0
song_id = song_id[song_id == True]
song_id = song_id.index
test_songs = songs.reset_index()
song_id = test_songs[test_songs["index"].isin(song_id) == True].index
y_pred = model(x_test[song_id])
pred = list(map(lambda x: np.argmax(x), y_pred))
actual = list(y_test[song_id])
acc = list(map(lambda x, y: x==y, pred, actual))
acc = acc.count(False) / len(acc)
print("False Positives in the test set: ", round(100 * acc, 2), "%")
print("Share of non-top songs in the training sample is ", round(100 * len(y_train[y_train == 0]) / len(y_train),2), "%")


#%% PAST POPULARITY (Model V: LSTM)

seq_dat = pred_dat
seq_dat = seq_dat.drop(seq_dat.columns[[0,1,2,16,17]], axis = 1)
seq_dat = seq_dat.values
seq_dat = seq_dat.reshape(seq_dat.shape[0] // 3, 3, seq_dat.shape[1]) # reshape to a tensor of (112 artists, 3 albums, 14 features album)
seq_dat.shape

# define train and test set
n_train = seq_dat.shape[0] // 5 * 4 # 80% for training, 20% for testing
train = seq_dat[:n_train]
test = seq_dat[n_train:]

# split into input and outcome
x_train, x_test = train[:, :, :-1], test[:, :, :-1] 
y_train, y_test = train[:, :, -1], test[:, :, -1]

# illustrate
print("shape of training input:", x_train.shape)
print("shape of training outcome:", y_train.shape)
print("shape of test input:", x_test.shape)
print("shape of test outcome:", y_test.shape)

# define the model -------------
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(10, # 10 neurons
                               input_shape=(x_train.shape[1], x_train.shape[2]),
                               return_sequences = True)) # indicate number time stamps and number of features
model.add(tf.keras.layers.Dense(2, activation="softmax"))
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train the model ---------------------------------
hist = model.fit(x_train, y_train, 
                 epochs=25, batch_size = 16,
                 validation_data=(x_test, y_test))

# evaluate overall accuracy -------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['accuracy'])
axs[1].plot(hist.epoch, hist.history['val_accuracy'])
axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()


# precision and false positive rate of newest song ---------------------
song_id = np.where(y_test[:,2] == 1)
song_id = song_id[0].tolist()
y_pred = model(x_test[song_id, :, :])
pred = list(map(lambda x: np.argmax(x), y_pred[:, 2, :])) # prediction for newest song
actual = list(y_test[song_id, 2])
acc = list(map(lambda x, y: x == y, pred, actual))
acc = acc.count(True) / len(acc)
print("Accuracy among top songs in the test set is: ", round(100 * acc, 2), "%")
print("Share of top songs in the training sample is ", round(100 * len(y_train[y_train[:, 2] == 1]) / len(y_train), 2), "%")

song_id = np.where(y_test[:,2] == 0)
song_id = song_id[0].tolist()
y_pred = model(x_test[song_id, :, :])
pred = list(map(lambda x: np.argmax(x), y_pred[:, 2, :])) # prediction for newest song
actual = list(y_test[song_id, 2])
acc = list(map(lambda x, y: x == y, pred, actual))
acc = acc.count(True) / len(acc)
print("False Positives in the test set: ", round(100 * (1- acc), 2), "%")
print("Share of non-top songs in the training sample is ", round(100 * len(y_train[y_train[:, 2] == 0]) / len(y_train), 2), "%")




