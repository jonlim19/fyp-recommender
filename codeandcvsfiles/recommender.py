import re
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tabulate import tabulate

all_df = pd.read_csv("2020-QS-World-University-Rankings_CLEANED.csv")
uni_df = pd.read_csv("uni_description.csv")
uni_keyword = pd.read_csv("university_keyword.csv")
university = pd.read_csv("keywords.csv")
all_uni = pd.read_csv("all_uni.csv")
all_uni2 = pd.read_csv("all_uni2.csv")
uni_keyword2 = pd.read_csv("university_keyword2.csv")

def initialize():
    newframe = all_df[['Institution Name','Country','Academic Reputation_SCORE','Employer Reputation_SCORE','Citations per Faculty_SCORE','Faculty Student_SCORE']]
    print(newframe)
    newframe.to_csv("university_keyword2.csv")

# Keyword Generator
def prepare():
    regex = re.compile('[%s]' % re.escape(string.punctuation)) # Regex for removing punctuation
    res = uni_df["desc"]
    stop_words = set(stopwords.words('english')) # Initialize stopwords
    lemmatizer = WordNetLemmatizer() # Initialize lemmatization
    all_keywords = []
    uni_key = {}
    for i in res.index:
        lowercase = res.iloc[i].lower()
        out = regex.sub('', lowercase)
        # print(str(i) + ": " + out)
        word_tokens = word_tokenize(out)

        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        # print(str(i) + ": " + str(filtered_sentence))
        lemm_list = []
        for word in filtered_sentence:
            lemm = lemmatizer.lemmatize(word)
            lemm_list.append(lemm)
        print(str(i) + ": " + str(lemm_list))
        all_keywords.append(lemm_list)
        uni_key[uni_keyword["Institution Name"].iloc[i]] = lemm_list
    # print(uni_key)
    keywords = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in uni_key.items()])).transpose()
    keywords = keywords.apply(lambda x: ','.join(x.dropna()), axis=1)
    keywords = pd.DataFrame(keywords)
    keywords.rename(columns={0: 'keywords'}, inplace=True)
    keywords.to_csv(path_or_buf='keywords.csv')
    # res.to_csv("university_keyword.csv")
    # dataframe = pd.DataFrame(all_keywords)
    # print(dataframe)


def join():
    res = pd.concat([uni_keyword2,university], axis=1)
    print(res['Country'])
    res.to_csv("all_uni2.csv")


def view():
    keywords = pd.read_csv('keywords.csv')
    # keywords.rename(columns={'Unnamed: 0': 'tconst'}, inplace=True)
    # keywords.set_index('tconst', inplace=True)
    uni = uni_keyword2.join(keywords,how="inner")
    uni.to_csv(path_or_buf='all_uni2.csv')
    print(uni[["Institution Name","Country","keywords"]])


def jaccard(uni1,uni2):
    a = set(uni1.split(','))
    b = set(uni2.split(','))
    numerator = a.intersection(b)
    denominator = a.union(b)

    return float(len(numerator)) / len(denominator)


def university_recommendation(university_name):
    university = all_uni.loc[all_uni["Institution Name"] == university_name]
    keywords = university["keywords"].iloc[0]
    tableformat = []
    jaccards = []
    for word in all_uni['keywords']:
        jaccards.append(jaccard(keywords, word))
    jaccards = pd.Series(jaccards)
    jaccards_index = jaccards.nlargest(6).index
    matches = all_uni2.loc[jaccards_index]

    print("Chosen university is : " + university_name + "\n")
    # Uncomment to see score
    # for match,score in zip(matches['Institution Name'][1:], jaccards[jaccards_index][1:]):
    #     print(match, score)
    for match,country,academic,employer,citation,student in zip(matches['Institution Name'], matches['Country'], matches['Academic Reputation_SCORE'],  matches['Employer Reputation_SCORE'],  matches['Citations per Faculty_SCORE'],  matches['Faculty Student_SCORE']):
        if match != university_name:
            tableformat.append([match,country,academic,employer,citation,student])
    print(tabulate(tableformat, headers=['University Name', 'Country', 'Academic Reputation', 'Employer Reputation', 'Research Output', 'Student Faculty'],tablefmt='orgtbl'))


if __name__ == '__main__':
    # initialize()
    # prepare()
    # join()
    # view()
    # scan()
    # print("Please enter your chosen University: ")
    user_choice = input("Please enter your chosen University: ")
    university_recommendation(user_choice)
