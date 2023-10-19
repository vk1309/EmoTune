# Music-Recommender-System

## 1. Introduction

Music has been a part of human life for thousands of years, and its impact on our emotions and
mental well-being is undeniable. Music can elicit powerful emotions, helping us manage our moods,
reduce stress, and promote overall well-being. However, finding the right music to match one's
emotional state can be a challenging and time-consuming task. An automated recommendation
system that suggests music based on the user's emotions could save users time and improve their
mood. 

This project presents a natural language processing (NLP) project that aims to develop such a
recommendation system using sentiment analysis, text classification, and machine learning
algorithms to analyze user input and recommend appropriate music.

Recent advances in NLP techniques, such as sentiment analysis, have made it possible to accurately
and efficiently identify the emotional tone of a piece of text, such as a social media post, review, or
message. 

In this project, the user is prompted to answer a few questions based on which the
sentiment of user through their input is analyzed, and then a recommendation system identifies the
user's emotional state and suggests appropriate music to match their mood. Text classification
techniques have also been used to categorize user input and suggest music that falls within specific
genres or styles. The success of the system will be evaluated based on its accuracy and effectiveness
of recommendations using metrics such as precision, recall, and F1 score.

The motivation behind this project is to leverage the power of NLP and machine learning algorithms
to create an automated music recommendation system that can improve the well-being of users,
enhance music discovery, and benefit the music industry as a whole. This system can be particularly
helpful for people who suffer from mental health problems or emotional volatility.

By suggesting music that aligns with the user's emotional state, the system can help users discover new artists and
genres and increase their overall consumption of music.

The project draws from a range of literature, including previous work on sentiment analysis
methodologies and techniques, such as the use of convolutional neural networks (CNNs) and recurrent neural networks (RNNs). The dataset used for training, validation, and testing includes sentences with corresponding emotion labels, such as sadness, anger, love, surprise, fear, and joy.

The recommendation system will analyze user input and suggest appropriate music that aligns with the user's emotional state, ultimately improving their well-being.

## 2. Dataset Description

The dataset has been obtained from kaggle, it is a collection of documents and its emotions. For
example: I feel like I am still looking at a blank canvas blank pieces of paper; sadness. The dataset
was prepared using semi-supervised, graph-based algorithm to produce rich structural descriptors
which serve as the building blocks for constructing contextualized affect representations from text.
The data has been split into training, validation, and test sets, with 80% of the data used for training,
10% for validation, and 10% for testing. The files ‘train.txt’, ‘val.txt’, and ‘test.txt’ contain the
sentences and their corresponding emotion labels. Each text file contains a sentence, along with the
corresponding emotion label, which is one of the following: 'sadness', 'anger', 'love', 'fear' and 'joy'.
So, for this project, there are a total of 5 emotional labels.

For data preparation to pass the text data into the machine learning models, the dataset has been
preprocessed to extract features. These features include the following:

1. Word count,
2. Character count,
3. Average word length,
4. Sentiment scores.
5. 
Sentimental analysis is done to calculate the sentiment scores. This information is used to recognize
emotions. It should be noted, however, that the dataset may not be indicative of feelings conveyed
in other languages or cultures, and the preprocessed features may not be adequate for some
machine learning tasks. These constraints should be considered when using the sample for machine
learning tasks.

<img width="369" alt="Screenshot 2023-10-19 at 9 28 54 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/fcb46abd-4efa-46b5-a878-79f200500a0b">

From the above plot, it can be observed that ‘sadness’ emotion has maximum records, crossing
5000, while joy on the other hand has the least number of records close to about 1500. 

<img width="369" alt="Screenshot 2023-10-19 at 9 29 30 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/b11169d6-3f88-42f8-bb97-441e20943127">

From the above plot, it can be observed that ‘sadness’ emotion has maximum records, nearing 700,
while ‘love’ on the other hand has the least number of records close to about 200. 

<img width="387" alt="Screenshot 2023-10-19 at 9 30 08 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/a09e58bd-6ca7-40a1-8305-2e7c4fccea5a">

From the above plot, it can be observed that ‘sadness’ emotion has maximum records, nearing 700,
while ‘fear’ on the other hand has the least number of records close to about 200. 

## 3. Methodology

The first step in developing the music recommendation system is to collect data from the user. The
user is prompted to answer a series of questions designed to gather information about their current
emotional state. The questions include inquiries about recent life events, daily routines, or favorite
activities. The answers provided by the user are then analyzed using sentiment analysis techniques
to determine the user's current emotional state. This methodology allows the system to accurately
classify the user's mood into one of the five basic emotions: anger, joy, fear, love, and sadness.

After determining the user's emotional state, the recommendation system uses the Spotify Web API
to extract and analyze the top 50 songs associated with the user's emotional state. The system also
extracts the top 5 genres associated with each of the five basic emotions to ensure a diverse range
of recommendations. This methodology is crucial to ensure that the system provides relevant music
recommendations to the user based on their current emotional state.

Finally, the recommendation system provides a list of songs and playlists that are most likely to
resonate with the user's current emotional state. The recommendations are based on the user's
current emotional state, listening history, and user-generated data.

<img width="590" alt="Screenshot 2023-10-19 at 9 31 45 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/4d8371fd-f6de-4e7e-903c-f9db0d883a70">

The above diagram depicts the interaction between frontend and backend. First the user is
prompted to fill an HTML form which takes in Client_id, Client_Secret, redirect_URL and 5 questions
which are used for emotion prediction. Once the user fills in the form a ‘POST’ API sends the details
to the backed where, based on the answers, the sentiment is analysed, an emotion is created and
songs are recommended based on the emotion. These songs are then pushed to fronted using ‘GET’
API. The architecture and implementation for the backend is explain below:

<img width="461" alt="Screenshot 2023-10-19 at 9 32 33 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/baa3a93b-68ce-4e08-9d7e-b7ad5d3969e9">

The codes are explained in detail below:

1) NLP_Project.ipynb:
   
This code starts with importing necessary libraries such as nltk, pandas, numpy, and keras. Then, the
dataset is loaded using pandas from three different files: train.txt, test.txt, and val.txt.
After loading the dataset, the data is preprocessed by removing HTML tags, URLs, numbers, stop
words, and performing stemming and lemmatization. Then, the data is split into train, test, and
validation sets.

Next, the target variable (Sentiment) is encoded using LabelEncoder and one-hot encoding
techniques. Then, the text data is converted into numerical format using Tokenizer and padded with
zeros to create a fixed length sequence.

Finally, a bidirectional LSTM model is built using the Keras library, which takes the input sequence
and returns the predicted sentiment. The model is compiled with an Adam optimizer and trained on
the training dataset. The model is evaluated on the test dataset and performance metrics such as
accuracy, confusion matrix, and classification report are calculated.
This code is picked and stored as ‘model.h5’.

3) recomm.ipynb:
   
This is a Python script that performs clustering analysis and a LightGBM classification model on the
dataset of music attributes. The script first reads in the dataset (data.csv and data_by_genre.csv),
drops duplicates, and visualizes the correlation between the attributes using a heatmap. The
heatmap showing the correlations among the variables is depicted below:

<img width="694" alt="Screenshot 2023-10-19 at 9 34 47 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/f9c8ce89-43e1-4594-9124-e5d41a474d44">

After visualizing the heatmap, K-Means clustering is performed on a subset of attributes and a label
column is added to the original dataset indicating the cluster to which each song belongs. Then a
LightGBM classification model is trained to predict the cluster label, and the accuracy of the model is
evaluated on a test set. The model score obtained is 0.988.

After this, clustering is performed on a different dataset of music attributes by genre, using the same
K-Means clustering algorithm and pipeline as before. A t-SNE dimensionality reduction is then
applied to visualize the clusters in 2D space, and the resulting plot is displayed using Plotly. 

Finally the clustering model is applied to the test dataset of music attributes and assigns each song a cluster
label. Then, it reads in a separate dataset of top 50 songs and applies the clustering model to the
song attributes to determine which clusters are overrepresented in the top songs.

A snippet of cluster along with the frequencies of songs it contains is shown below:

<img width="398" alt="Screenshot 2023-10-19 at 9 35 35 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/45acd8e5-0cc0-4d28-94b6-9a30ca97a56a">

From the above figure, we observe that cluster 7 has maximum number of songs that are 14. On the
other hand, cluster 3 and cluster 8 have minimum number of songs that is 1. This model is pickled as
‘finalized_model.pkl’

3) top_50_rec.py:
   
This script starts by importing necessary libraries such as Spotipy, pandas, defaultdict, numpy, ast,
pickle and more_itertools. Spotipy is used to extract audio features and genres of songs from the
Spotify platform. The client_id, client_secret (secret key of client) and redirect_url is passed as
inputs to the code. To obtain these details, a developer’s account was created on Spotify (labelled as
‘Testgenre’ in our example). The screenshots of the sample ‘Testgenre’ and other credentials are
shown below:

<img width="628" alt="Screenshot 2023-10-19 at 9 36 30 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/60984d79-3a36-46d6-be42-864afd72b70f">

<img width="778" alt="Screenshot 2023-10-19 at 9 37 51 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/e3e85a22-5312-48aa-9eb3-2f1d5180097f">

The above figures show how a test application ‘Testgenre’ is created and visualized on dashboard.
Once the application is created, the credentials for client_id and client_secret and redirect_url are
obtained as depicted in the figures below:

<img width="780" alt="Screenshot 2023-10-19 at 9 38 33 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/be004e3b-9424-4d56-89d2-c635caa56f3a">

The ‘Client ID’, ‘Client secret’ and ‘Redirect URIs’ are passed in the code.

Pandas and defaultdict are used to manipulate data and dictionaries. Numpy is used to perform
mathematical operations on arrays, and ast is used to convert string literals to their corresponding
Python data types. More_itertools is used to remove duplicates from lists.

Next, the code defines a dictionary of five emotions ('anger', 'joy', 'fear', 'love', and 'sadness') and
their corresponding top five genres. It then uses the Spotipy library to search for songs matching
these genres, and appends the song results to a list.

The song list is then divided into sub-lists of 50 songs each, and the Spotipy library is used again to
extract audio features of each song. The audio features are stored in a nested list format. A nested
dictionary is then created to organize the extracted features by emotion category.

The extracted features are then stacked, and pre-trained clustering model is used to cluster the
songs into different emotion categories. The model predicts a cluster label for each song based on
their extracted audio features. The clustered results are then sorted and duplicates are removed.
Finally, a dictionary is created to store the emotion categories and their corresponding song cluster
IDs.

Lastly, the dictionary is pickled and saved as a file on disk. This output file contains the top genreemotion tracks clustered by emotion categories.

4) sent.py:
5) 
The code starts by importing the required libraries such as Tensorflow, Keras, NLTK (Natural
Language Toolkit), and Numpy. NLTK is used for natural language processing tasks such as
tokenization and stemming. The code then downloads the required NLTK data for tokenization and
lemmatization.

Next, the code defines the maximum number of words to be considered for tokenization and loads
the pre-trained sentiment analysis model using the keras.models.load_model function. The input X is
a string of text that needs to be classified into one of the five sentiments - anger, joy, fear, love, or
sadness.

The code then defines a preprocess function that preprocesses the input text by removing HTML
tags, URLs, numbers, and stop words using regular expressions and NLTK functions. The function
tokenizes the text using the RegexpTokenizer function and then applies stemming and
lemmatization using PorterStemmer and WordNetLemmatizer respectively. Finally, the function
returns the preprocessed text as a string.

The result function takes the preprocessed text as input and uses the pre-trained model to predict
the sentiment of the text. The texts_to_sequences function of the tokenizer object is used to
convert the preprocessed text into a sequence of integers. The pad_sequences function is then used
to ensure that all input sequences have the same length. The maximum length is set to 300 and any
sequence shorter than 300 is padded with zeros. The model.predict function is used to predict the
sentiment of the input text.

Finally, the sentiment predicted by the model is returned as an output. The sentiment is one of the
five values defined in the direct list - anger, joy, fear, love, or sadness. The output sentiment can be
used to perform further analysis or make decisions based on the sentiment of the input text.

5) spot_rec.py:
   
This is a python script that connects to the Spotify API and retrieves a set of recommended songs
based on a given sentiment. The script uses the Spotipy library, which is a Python wrapper for the
Spotify API, to authenticate with the user's Spotify account and request song recommendations
based on a set of seed tracks.

The script first imports the necessary libraries, including Spotipy, SpotifyOAuth, and Pandas. The
SpotifyOAuth library is used to authenticate with the Spotify API using the user's Spotify account.
Pandas is used to store and manipulate the retrieved data.

The script then defines a function called "get_output" that takes in four parameters: the client ID
and secret, the redirect URI, and the sentiment. The sentiment is used to retrieve a set of songs from
the Mender library, which is a library that clusters songs based on sentiment.

The function then initializes a Spotipy client with the given authentication parameters and retrieves
a set of recommended songs based on the seed tracks retrieved from the Mender library. The
function loops through the recommended tracks, retrieves the track name, artist name, track URL,
and album art URL for each track, and stores them in separate lists.

The function then loops through the artist names and retrieves the corresponding artist object from
the Spotify API. It then retrieves the album art URL for each artist and stores it in a separate list.
Finally, the function creates a Pandas DataFrame to store the retrieved data and returns it as the
function output.

The script then calls the "get_output" function with the given authentication parameters and stores
the retrieved data in a variable called "fin".

6) mender.py:

This code defines a function called cluster(x). The function loads a trained machine learning model
from a saved file using the pickle module, along with two datasets. It then uses the loaded model to
predict the cluster of a given input x and returns the top 5 songs from the corresponding cluster
based on the pre-defined emotion_dict.

A pickle.load() function reads a saved model from a file called "finalized_model_1.pkl" and the
datasets are read from "my_DEV" and "fin_uris" CSV files respectively. The pd.read_csv() function is
used to read the CSV files into Pandas dataframes. The loaded model is then used to predict the
cluster for the given input x using the predict() method. The cluster number is assigned to the
'cluster' column of the data_test dataframe.

The dataframe is then filtered by selecting only the columns related to the predicted cluster, i.e.,
'cluster' and 'uri', and stored in the data_fin variable. The top 5 songs corresponding to the predicted
cluster are selected using the pre-defined emotion_dict and stored in the seed_songs variable.
Finally, the function returns the top 5 songs from the cluster.

Overall, this function is used for recommending songs based on the predicted cluster of a given
input. The clusters are generated based on the pre-defined emotion_dict which maps the emotions
to the corresponding clusters. This allows for a personalized song recommendation system based on
the user's emotional state.

7) user_DEV_uri.py:

This script imports three libraries: "spotipy", "pandas", and "SpotifyOAuth". "spotipy" is a Python
library that provides easy access to the Spotify Web API. "pandas" is a library used for data
manipulation and analysis. "SpotifyOAuth" is a class from the Spotipy library that provides an
authorization flow for the Spotify Web API.

Next, a function "get_user_dev" is defined. This function takes in three parameters: "client_id",
"client_secret", and "redirect_uri". These parameters are used to authenticate the user and
authorize access to the Spotify Web API.

An instance of the SpotifyOAuth class is created, passing in the provided parameters. This is done to
get an access token for the user.

The variable "ranges" is defined as a list containing the string "short_term". "short_term" is a time
range used to retrieve the user's top tracks.
An empty list "uris" is created to hold the URI (Uniform Resource Identifier) of each of the user's top
tracks.

A loop is used to iterate over the "ranges" list. Inside the loop, the method
"current_user_top_tracks" is called to retrieve the user's top 50 tracks from the specified time range
(in this case, "short_term"). The URI of each track is extracted from the response and added to the
"uris" list.

The "uris" list is then converted to a pandas DataFrame object and saved to a CSV file named
"fin_uris". This file contains the URI of each of the user's top 50 tracks.

Next, a list of audio features to extract is defined as "features" and an empty list "feature" is defined
to hold the feature values of each track.

The "audio_features" method of the Spotipy object "sp" is called with the "uris" list as an argument
to retrieve the audio features of each track. The result is stored in the "feature" list.

A loop is used to extract the specified features from each track's feature data and append them to a
new list "final". The "final" list is then converted to a pandas DataFrame object and saved to a CSV
file named "my_DEV". This file will contain the danceability, energy, and valence audio features of
each of the user's top 50 tracks.

In summary, the script retrieves the user's top 50 songs, extracts the danceability, energy, and
valence audio features of each song, and saves the results to CSV files. The script uses the Spotipy
library to access the Spotify Web API and the pandas library for data manipulation and analysis. The
SpotifyOAuth class is used to authenticate the user and authorize access to the Spotify Web API.

## 4. Results

Below are test cases that show the results for ‘happy’ and ‘sad’ emotions:

<img width="589" alt="Screenshot 2023-10-19 at 9 41 57 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/6304bc5a-417f-4cf0-82d8-4a01bcd280f9">

This is the HTML form which inputs the user details and ask questions to be analysed for emotion
detection. Our algorithm predicted ‘happy’ as the emotion for the answers given by the user and
generated a list of 10 happy songs as below:

<img width="578" alt="Screenshot 2023-10-19 at 9 42 37 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/a659296b-52d6-4fa1-acc9-15f079effcce">

Similarly, for below are the representations for an emotion that was predicted ‘sad’:

<img width="518" alt="Screenshot 2023-10-19 at 9 43 15 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/ce9b115b-287e-4846-bcb1-454feef360cc">


<img width="547" alt="Screenshot 2023-10-19 at 9 43 42 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/1f6f51d8-eab6-4604-abc8-73a1583d0dbb">


<img width="547" alt="Screenshot 2023-10-19 at 9 44 26 AM" src="https://github.com/vk1309/EmoTune/assets/39329373/c44228e5-2485-4bee-9065-aa4a3cd87914">

From the above classification we can gauge that precision and recall are really good for all the
sentiments. Higher the precision signifies that our model has lesser values of false positives. This
means that our model has made a lot of accurate predictions and very few false positive predictions.
Similarly, a higher recall signifies that our model has lesser values of false negatives. We can clearly
see that recall too is really good for our model, which means that are very low false negative
predictions. The recall and precision are comparatively lower for ‘3’ because of lesser support. As
there are fewer records compared to other emotions, the precision and recall are low. To overcome
this, more records of emotions pertaining to class ‘3’ can be added.

## 5. Future Works:

While we did build a good recommendation system based on the emotion of the user, we still have
to productionize it. Right now, the model works perfectly on local machine, the next steps are to
host the website and make it global so that it reaches more users. Another enhancement that can be
made is adding another feature of ‘tempo’ while coding up the recommendation system. This would
make the song suggestions more personalized. Finally, a playlist with an option to ‘save’ can be
made rather than just recommending 10 songs to the user.

## 6. References: 

[1] Cambria, E., & Hussain, A. (2012). Sentic computing: Techniques, tools, and applications. Springer
Science & Business Media.

[2] Pang, B., & Lee, L. (2008). Sentiment Analysis and Opinion Mining. Synthesis Lectures on Human
Language Technologies, 1(1), 1-167.

[3] Yoon, K. (2014). A sensitivity analysis of (and practitioners' guide to) Convolutional Neural
Networks for Sentiment Analysis.

[4] Chang, W.-C., & Lin, C.-J. (2017). Recurrent neural network for text classification with multi-task
learning. arXiv preprint arXiv:1705.06114.

[5] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I.
(2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-
6008).

[6] Saravia, E. et al. (2018) “Carer: Contextualized affect representations for emotion recognition,”
Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.

[7] Kong X, Jiang H, Yang Z, Xu Z, Xia F, Tolba A (2016) Exploiting Publication Contents and
Collaboration Networks for Collaborator Recommendation. PLoS ONE 11(2): e0148492.

[8] Dongjoo Lee et al, Opinion Mining of Customer Feedback Data on the Web. Seoul National
University.

[9] Y. Koren, R. Bell, C. Volinsky. (2009) “Matrix Factorization Techniques for Recommender Systems”
Published in Computer.

[10] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional
transformers for language understanding.

[11] Joshi, A. and Popoola, O. (2021) “Classifying Emotions in Real-Time .” Stanford University.






















