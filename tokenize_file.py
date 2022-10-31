import re
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()


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


stop_words = set(stopwords.words('english'))

filename = 'files/text_1.txt'

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

vectors = vectorizer.fit_transform(word_tokens)

# Print how many rows and columns of the TF-IDF matrix consists
print("n_samples: %d, n_features: %d" % vectors.shape)

# Select the first five documents from the data set
tf_idf = pd.DataFrame(vectors.todense()).iloc[:5]
tf_idf.columns = vectorizer.get_feature_names_out()
tfidf_matrix = tf_idf.T
tfidf_matrix.columns = ['response'+str(i) for i in range(1, 6)]
tfidf_matrix['count'] = tfidf_matrix.sum(axis=1)

# Top 10 words
tfidf_matrix = tfidf_matrix.sort_values(by='count', ascending=False)

# Print the first 10 words
print(tfidf_matrix.drop(columns=['count']))
