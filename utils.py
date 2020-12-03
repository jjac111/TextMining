from config import *
import pandas as pd
import numpy as np
import os
from math import log
from matplotlib import pyplot as plt
from unidecode import unidecode
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


def euclidean_distance(vec1, vec2):
    assert len(vec1) == len(vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    normalized_1 = [0 for i in range(len(vec1))]
    normalized_2 = [0 for i in range(len(vec2))]

    if norm1:
        normalized_1 = vec1 / norm1
    if norm2:
        normalized_2 = vec2 / norm2

    return np.dot(normalized_1, normalized_2)


def vectorize(vocabulary, tokenized, tf_idf, tweet_id=None, num_documents=0):
    vector = [0 for i in range(len(vocabulary))]
    for word in tokenized:
        if word not in vocabulary:
            continue
        idx = vocabulary.index(word)
        if tweet_id:
            tf_idf_ = tf_idf[tf_idf['tweet_id'] == tweet_id]
            tf_idf_ = tf_idf_[tf_idf_['word'] == word]['tf-idf']
            if type(tf_idf_) != int:
                if tf_idf_.empty:
                    continue
                tf_idf_ = tf_idf_.iloc[0]
            vector[idx] += tf_idf_
        else:
            df = tf_idf[tf_idf['word'] == word]['df']
            if type(df) != int:
                if df.empty:
                    continue
                df = df.iloc[0]

            idf = log(num_documents / df) if df else 0
            vector[idx] += idf

    return vector


def tokenize(text):
    words = text.lower().split()
    words = [word for word in words if word not in stopwords and len(word) > 2 and word.isalpha()]

    return words


def top_candidates(df, only_manifestos):
    candidates_usernames = pd.read_csv(candidates_csv_file)['twitter_screen_name']
    candidates_lastnames = pd.read_csv(candidates_csv_file)['apellidos']
    candidates_lastnames = {u: unidecode(s.split()[0]).lower() for u, s in
                            zip(candidates_usernames, candidates_lastnames)}

    if only_manifestos:
        candidates_lastnames_to_users = {ln: u for u, ln in candidates_lastnames.items()}

        existing_manifestos_usernames = []
        for file in os.listdir(manifesto_data_directory):
            lastname = file.split('.')[0]
            existing_manifestos_usernames.append(candidates_lastnames_to_users[lastname])

        candidates_usernames = candidates_usernames[candidates_usernames.isin(existing_manifestos_usernames)]

    candidates_tweets = df[df['tweet_screen_name'].isin(candidates_usernames)].sample(frac=tweet_fraction)
    # mentioned_tweets = df[df['tweet_text'].map(lambda x: any(y in x for y in candidates_usernames))]

    print(f'Total tweets: {len(df)}')
    print(f'Total tweets from candidates: {len(candidates_tweets)}')
    # print(f'Total tweets with mentioned candidates: {len(mentioned_tweets)}\n')

    candidates = candidates_tweets.groupby('tweet_screen_name').count().sort_values('tweet_id', ascending=False)
    candidates = candidates.rename(columns={'tweet_id': 'count'}).head(num_candidates)['count']

    print(f'The top {num_candidates} most active candidates are:')
    print(candidates)

    candidates_tweets = candidates_tweets[candidates_tweets['tweet_screen_name'].isin(candidates.index.tolist())]

    return candidates_tweets, candidates_lastnames


def top_words(candidates_tweets):
    candidates_words = {}
    vocabulary = {}

    for i, row in candidates_tweets.iterrows():
        user = row['tweet_screen_name']
        tweet = row['tweet_text']
        date = row['tweet_date']

        candidates_words[user] = candidates_words.get(user, {})
        tokenized = tokenize(tweet)

        for word in tokenized:
            candidates_words[user][word] = candidates_words[user].get(word, [])
            candidates_words[user][word].append(date)
            vocabulary[word] = vocabulary.get(word, 0) + 1

    print(f'\nThe entire vocabulary has size {len(vocabulary)}\n')

    top = {}
    top_words_set = []
    tf_idf = pd.DataFrame(columns=['word', 'tweet_id', 'tf', 'df', 'idf', 'tf-idf'])
    print('Top 10 words for candidates:')
    for user in candidates_words.keys():
        words = dict(sorted(candidates_words[user].items(), key=lambda x: len(x[1]), reverse=True)[:10])
        top[user] = words
        word_list = list(top[user].keys())
        top_words_set.extend(word_list)
        print(f'{user}\t{word_list}\n')

    top_words_set = set(top_words_set)

    for word in top_words_set:
        df_ = sum([1 for doc in candidates_tweets['tweet_text'] if word in doc])
        idf_ = log(len(candidates_tweets) / df_) if df_ else 0

        for i, row in candidates_tweets.iterrows():
            tweet_id, tweet = row['tweet_id'], row['tweet_text']
            tf_ = tweet.count(word)
            tf_idf_ = tf_ * idf_

            tf_idf = tf_idf.append(
                {'word': word, 'tweet_id': tweet_id, 'tf': tf_, 'df': df_, 'idf': idf_,
                 'tf-idf': tf_idf_},
                ignore_index=True)

    return top, tf_idf, list(sorted(vocabulary.keys())), top_words_set


def time_line(candidates_words, candidates_tweets):
    to_plot = {}
    for user, words in candidates_words.items():
        max_week = max(candidates_tweets[candidates_tweets['tweet_screen_name'] == user]['tweet_date']).isocalendar()[1]
        min_week = min(candidates_tweets[candidates_tweets['tweet_screen_name'] == user]['tweet_date']).isocalendar()[1]
        to_plot[user] = {}
        plt.figure()

        for word, dates in words.items():
            to_plot[user][word] = {}
            for week in [i for i in range(min_week, max_week + 1)]:
                to_plot[user][word][week] = 0

            for date in dates:
                week = date.isocalendar()[1]
                to_plot[user][word][week] += 1

            plt.plot(list(to_plot[user][word].keys()), list(to_plot[user][word].values()), label=word)
        plt.legend()
        plt.grid()
        plt.title(f'Word usage of user: {user}')
        plt.xlabel('Week of the year')
        plt.ylabel('Word count')
        plt.show()


def ranking(query, vocabulary, candidates_tweets, tf_idf):
    query_vector = vectorize(vocabulary, tokenize(query), tf_idf, num_documents=len(candidates_tweets))
    print(f'The query vector is:\n{query_vector}\n')

    scores = {}
    tweet_total_vectors = {}
    for i, row in candidates_tweets.iterrows():
        user = row['tweet_screen_name']
        tweet = row['tweet_text']
        tweet_id = row['tweet_id']

        tweet_vector = vectorize(vocabulary, tokenize(tweet), tf_idf, tweet_id=tweet_id)

        if not tweet_total_vectors.get(user):
            tweet_total_vectors[user] = tweet_vector
        else:
            tweet_total_vectors[user] = [x + y for x, y in zip(tweet_total_vectors[user], tweet_vector)]

        user_tweets = scores.get(user, {})
        user_tweets[tweet_id] = euclidean_distance(query_vector, tweet_vector)
        scores[user] = user_tweets

    scores = list(sorted(scores.items(), key=lambda x: sum(x[1].values()), reverse=True))

    for i, (candidate, tweets) in enumerate(scores):
        print(f'{i + 1}.\t{candidate}')
        print(f'Total score:    {sum(tweets.values())}')
        top_tweet_id = max(tweets, key=tweets.get)
        print(f'Top ranked tweet:')
        print(candidates_tweets[candidates_tweets['tweet_id'] == top_tweet_id]['tweet_text'].values[0])
        top_tweet_score = tweets[top_tweet_id]
        print(f'Tweet score: {top_tweet_score}\n')

    return tweet_total_vectors


def similarity(total_user_vectors):
    similarities = pd.DataFrame(columns=list(total_user_vectors.keys()), index=list(total_user_vectors.keys()))

    for user_1, vec_1 in total_user_vectors.items():
        for user_2, vec_2 in total_user_vectors.items():
            sim = euclidean_distance(vec_1, vec_2)
            similarities.loc[user_1][user_2] = sim

    return similarities


def deviation(candidates_words, candidates_lastnames, vocabulary, candidates_tweets, total_user_vectors, tf_idf):
    manifesto_vectors = {}
    for user in candidates_words.keys():
        lastname = candidates_lastnames[user]

        output_string = StringIO()

        try:
            ##########################################################################
            with open(manifesto_data_directory + lastname + '.pdf', 'rb') as in_file:
                parser = PDFParser(in_file)
                doc = PDFDocument(parser)
                rsrcmgr = PDFResourceManager()
                device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                for page in PDFPage.create_pages(doc):
                    interpreter.process_page(page)

            text = output_string.getvalue().replace('•', ' ').replace('\n', ' ').replace('\x0c', ' ')
            #########################################################################
            # Code copied from pdfminer.six documentation:
            # https://pdfminersix.readthedocs.io/en/latest/tutorial/composable.html

            manifesto_vectors[user] = vectorize(vocabulary, tokenize(text), tf_idf,
                                                num_documents=len(candidates_tweets))


        except Exception as e:
            print(f'Manifesto for {lastname} not found.')

    deviations = {}
    for user, manifesto_vector in manifesto_vectors.items():
        manifesto_vector_reduced = []
        tweet_total_vector_reduced = []
        for word in candidates_words[user]:
            idx = vocabulary.index(word)
            manifesto_vector_reduced.append(manifesto_vector[idx])
            tweet_total_vector_reduced.append(total_user_vectors[user][idx])

        deviations[user] = euclidean_distance(manifesto_vector_reduced, tweet_total_vector_reduced)

    plt.figure()
    plt.bar(deviations.keys(), deviations.values())
    plt.title('Euclidean distance of reduced vocabulary vectors of tweets vs. manifestos')
    plt.xlabel('Candidates')
    plt.ylabel('Euclidean distance')
    plt.xticks(rotation=60)
    plt.show()

