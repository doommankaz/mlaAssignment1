import re
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

# POS_TAGGER_FUNCTION : TYPE 1
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def format_file(filename):
    stop_words = set(stopwords.words('english'))

    file = open(filename, 'r')  # opens a filename to read-only
    text = file.read()  # create a variable to write our text to
    text = text.lower()  # text to lowercase
    text = text.replace('\n', ' ')  # remove all enters
    text = re.sub(r"\d+", "", text, flags=re.UNICODE)  # remove all numbers
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)  # remove all punctuation marks

    word_tokens = word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

    wnl = WordNetLemmatizer()

    pos_tagged = nltk.pos_tag(filtered_sentence)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    lmt_sentence = []

    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lmt_sentence.append(word)
        else:
            # else use the tag to lmt the token
            lmt_sentence.append(wnl.lemmatize(word, tag))
    lmt_sentence = " ".join(lmt_sentence)

    word_tokens = word_tokenize(lmt_sentence)
    print(word_tokens)

    return lmt_sentence


def computetfidf(wordarray):
    # instantiate the vectori
    # zer object
    countvectorizer = CountVectorizer(analyzer='word', stop_words='english')
    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')

    # convert th documents into a matrix
    count_wm = countvectorizer.fit_transform(wordarray)
    tfidf_wm = tfidfvectorizer.fit_transform(wordarray)

    count_tokens = countvectorizer.get_feature_names_out()
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    df_countvect = pd.DataFrame(data=count_wm.toarray(), index=['Doc1', 'Doc2', 'Doc3', 'Doc4', 'Doc5', 'Doc6', 'Doc7',
                                                                'Doc8', 'Doc9', 'Doc10'], columns=count_tokens)
    df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), index=['Doc1', 'Doc2', 'Doc3', 'Doc4', 'Doc5', 'Doc6', 'Doc7',
                                                                'Doc8', 'Doc9', 'Doc10'], columns=tfidf_tokens)
    print("Count Vectorizer\n")
    print(df_countvect)
    print("\nTD-IDF Vectorizer\n")
    print(df_tfidfvect)


if __name__ == '__main__':
    wordlist = []
    for i in range(1, 11):
        wordlist.append(format_file('files/text_' + str(i) + '.txt'))
    computetfidf(wordlist)
