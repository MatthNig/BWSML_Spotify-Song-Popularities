library("spotifyr")
library("dplyr")

# specify authenticication key:
client_id <- 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx' # set your client_id here
Sys.setenv(SPOTIFY_CLIENT_ID = client_id)
Sys.setenv(SPOTIFY_CLIENT_SECRET = client_id)
access_token <- get_spotify_access_token()

# load artist names from kaggle dataset:
artist_album <- read.csv(paste(getwd(), "/artist_album.csv", sep = ""), stringsAsFactors = FALSE)
artists <- as.character(unique(artist_album$artist_name))

##############################################
##### Get album release date information #####
##############################################

release_date_fun <- function(artist){
        
        songs <- as.character(artist_album[artist_album$artist_name == artist,"song_name"])
        songs <- tolower(songs)
        
        scrap_fun <- function(artist){
                df <- get_artist_audio_features(artist = artist, include_groups = "album")
                df$track_name <- tolower(df$track_name)
                df <- subset(df, track_name %in% songs)
                df <- df[!duplicated(df$track_name),]
                df <- select(df, artist_name, album_name, album_release_date, album_release_year, track_name)
                df <- rename(df, song_name = track_name, album_names = album_name)
                return(df)
        }
        
        df <- tryCatch(scrap_fun(artist = artist), error=function(e) data.frame(artist_name = NA,
                                                                                album_names = NA,
                                                                                album_release_date = NA,
                                                                                album_release_year = NA,
                                                                                song_name = NA))
        return(df)
}

# ---------------------- NOT RUN -------------------------------------
# get album release information of all songs from kaggle dataset:
                       
# artist_list <- lapply(artists, function(x) release_date_fun(x))
# artist_list <- bind_rows(artist_list)


#########################
##### Illustrations #####
#########################

# Newly released songs ----------------------------------------------------------

# Pearl Jam
tmp <- get_artist_audio_features(artist = "pearl jam", include_groups = "album")
vars <- c('track_name', 'duration_ms', 'acousticness',
          'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
          'loudness', 'mode', 'speechiness', 'tempo', 'time_signature',
          'valence')
df <- tmp[tmp$album_name == "Gigaton", vars][1,]

# Nine Inch Nails
tmp <- get_artist_audio_features(artist = "Nine Inch Nails", include_groups = "album")
vars <- c('track_name', 'duration_ms', 'acousticness',
          'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
          'loudness', 'mode', 'speechiness', 'tempo', 'time_signature',
          'valence')
df <- rbind(df, tmp[1, vars])

# The Weekend
tmp <- get_artist_audio_features(artist = "The Weekend", include_groups = "album")
vars <- c('track_name', 'duration_ms', 'acousticness',
          'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
          'loudness', 'mode', 'speechiness', 'tempo', 'time_signature',
          'valence')
df <- rbind(df, tmp[1, vars])

# Dua Lipa
tmp <- get_artist_audio_features(artist = "Dua Lipa", include_groups = "album")
vars <- c('track_name', 'duration_ms', 'acousticness',
          'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
          'loudness', 'mode', 'speechiness', 'tempo', 'time_signature',
          'valence')
df <- rbind(df, tmp[1, vars])

# Further examples ----------------------------------------------------------

# Katy Perry - Firework
tmp <- get_artist_audio_features(artist = "Katy Perry", include_groups = "album")
tmp <- tmp[tmp$track_name == "Firework",]
vars <- c('track_name', 'duration_ms', 'acousticness',
          'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
          'loudness', 'mode', 'speechiness', 'tempo', 'time_signature',
          'valence')
df <- rbind(df, tmp[1, vars])

# Rolling Stones - Angie
tmp <- get_artist_audio_features(artist = "Rolling Stones", include_groups = "album")
tmp <- tmp[tmp$track_name == "Angie - Remastered",]
vars <- c('track_name', 'duration_ms', 'acousticness',
          'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
          'loudness', 'mode', 'speechiness', 'tempo', 'time_signature',
          'valence')
df <- rbind(df, tmp[1, vars])

# Lady Gaga
tmp <- get_artist_audio_features(artist = "Lady Gaga", include_groups = "album")
tmp <- tmp[tmp$track_name == "Poker Face",]
vars <- c('track_name', 'duration_ms', 'acousticness',
          'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
          'loudness', 'mode', 'speechiness', 'tempo', 'time_signature',
          'valence')
df <- rbind(df, tmp[1, vars])


df$artist_name <- c("Pearl Jam", "Nine Inch Nails", "The Weekend", "Dua Lipa", "Katy Perry",
                    "Rolling Stones", "Lady Gaga")

                       
# Check structure ----------------------------------------------------------
# df ordering copied from .py code structure
# nT<-c('song_name', 'song_popularity', 'song_duration_ms', 'acousticness',
#      'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
#      'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
#      'audio_valence', 'song_pop_class', 'top_song')

# nT <- nT[-c(2,16,17)]
tmp <- c("song_name","song_duration_ms","acousticness","danceability",
              "energy", "instrumentalness", "key", "liveness", "loudness",
              "audio_mode", "speechiness", "tempo", "time_signature", "audio_valence")
# tmp == nT
names(df) <- tmp
