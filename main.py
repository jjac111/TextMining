from config import *
from utils import *
import pandas as pd

df = pd.read_pickle(pkl_file)

candidates_tweets, candidates_lastnames = top_candidates(df, only_manifestos=True)

candidates_words, tf_idf, vocabulary, top_words_set = top_words(candidates_tweets)

time_line(candidates_words, candidates_tweets)

total_user_vectors = ranking('alcaldía quito ciudad trabajamos empresa', vocabulary, candidates_tweets, tf_idf)

print(similarity(total_user_vectors))

deviation(candidates_words, candidates_lastnames,  vocabulary, candidates_tweets, total_user_vectors, tf_idf)

