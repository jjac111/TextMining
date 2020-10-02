from nltk.corpus import stopwords

twitter_data_directory = r'C:\Users\JuanJavier\Downloads\2019-SectionalElections\2019-SectionalElections\01_TwitterDB\\'
manifesto_data_directory = r'C:\Users\JuanJavier\Downloads\2019-SectionalElections\2019-SectionalElections' \
                           r'\00_Manifestos\00_Quito\\'
pkl_file = twitter_data_directory + r'obj_livetweets_pandasdataframe.pkl'
candidates_csv_file = twitter_data_directory + r'account_info.csv'

stopwords = stopwords.words('spanish')

num_candidates = 5
tweet_fraction = 0.1
