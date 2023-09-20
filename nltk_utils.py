import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words_all):
    """
        return bag of words array:
        1 for each known word that exists in the sentence, 0 otherwise
        example:
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
        """
    # stem each word
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words_all), dtype=np.float32)
    for idx, w in enumerate(words_all): #token in the bag w that index
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag

#try the function --------------
#sentence = ["hello", "how", "are", "you"]
#words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
#bag = bag_of_words(sentence, words)
#print(bag)

a = "How old are you?"
print(a)
a = tokenize(a)
print(a)

words = ["Run", "runs", "running"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)