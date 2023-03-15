import nltk
from nltk.corpus import stopwords
from nltk.lm.preprocessing import flatten, pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE
from nltk.util import bigrams, trigrams
from nltk.util import everygrams
from nltk.lm import KneserNeyInterpolated, Laplace, StupidBackoff, AbsoluteDiscountingInterpolated

import string, sys, math


def data_preprocessing(name, test):
    # Declare necessary variables 
    clean_words = {}
    
    sentences = []
    ftemp = open(name[:-1], 'r', encoding='utf-8')
    temp_lines = ftemp.readlines()
    real_lines = ""
    for line in temp_lines:
        real_lines += line
    sentences = real_lines.split('.')

    # Clean the data 
    table = str.maketrans('', '', string.punctuation)
    stripped = []
    clean_sentences = []
    for sentence in sentences[:-1]:
        stripped = sentence.translate(table)
        lower = stripped.lower()
        individual_sentence = lower.split()

        # Remove non-alphabetic strings (i.e. punctuation) from the data
        clean_sentence = []
        for word in individual_sentence:
            alpha = "" 
            for index in word:
                if index.isalpha():
                    alpha += index
            if alpha != "":
                clean_sentence += [alpha]
        clean_sentences += [clean_sentence]

    ftemp.close()

    # If test flag is present, then we want to train model on all data, if not, we need to break data into a training and dev set.
    if test:
        train, vocab = padded_everygram_pipeline(3, clean_sentences)
        return [(train, vocab), []]
    else:
        # Break data into train and dev set
        clean_length = len(clean_sentences)//10
        train_set = clean_sentences[clean_length+1:]

        #train model on train set
        train, vocab = padded_everygram_pipeline(3, train_set)

        #separate dev set and create everygrams based on each sentence
        dev_set = clean_sentences[:clean_length]
        every = []
        for sentence in dev_set:
            padded_bigrams = list(pad_both_ends(sentence, n=2))
            every += [list(bigrams(padded_bigrams))]
        return [(train, vocab), every]


def test_data_processing(name):
    # Declare necessary variables 

    ftemp = open(name, 'r', encoding='utf-8')
    temp_lines = ftemp.readlines()
    table = str.maketrans('', '', string.punctuation)
    clean_sentences = []
    every = []

    for line in temp_lines:
        stripped = []

        # Clean the data 
        stripped = line.translate(table)
        lower = stripped.lower()
        individual_sentence = lower.split()

        # Remove non-alphabetic strings (i.e. punctuation) from the data
        clean_sentence = []
        for word in individual_sentence:
            alpha = "" 
            for index in word:
                if index.isalpha():
                    alpha += index
            if alpha != "":
                clean_sentence += [alpha]
        clean_sentences += [clean_sentence]
    
    for sentence in clean_sentences:
        padded_bigrams = list(pad_both_ends(sentence, n=2))
        every += [list(bigrams(padded_bigrams))]

    ftemp.close()
    return every 

# Our data has been processed and cleaned, now we train our model
def model_training(model, order, smoothing):
    if smoothing == 'k':
        lm = KneserNeyInterpolated(order = order)
    elif smoothing == 'a':
        lm = AbsoluteDiscountingInterpolated(order = order)
    else:
        lm = Laplace(order=order)
    train = model[0]
    vocab = model[1]
    lm.fit(train, vocab)

    return lm
# Function to use the model
def use_model(model, sequence):
    return model.perplexity(sequence)

# Function to use trained model to generate text
def generate(model):
    return model.generate(20, text_seed=['i'])


# Check for command line arguments. If there is more than just an authorlist present, then check if the second argument is the test flag. If it is, train the model fully on all of the data in the authorfile and output the classification result for each line in the testfile.
if len(sys.argv) == 2:
    f = open(sys.argv[1], "r")
    file_names = f.readlines()
    print("List of files we will be reading in: ", file_names)

    models = {}
    dev_data = {}
    for text in file_names:
        data_package = data_preprocessing(text, False)
        if text[:-1] == 'dickens_utf8.txt':
            models[text[:-1]] = model_training(data_package[0], 2, 'a')
        elif text[:-1] == 'wilde_utf8.txt':
            models[text[:-1]] = model_training(data_package[0], 2, 'a')
        else:
            models[text[:-1]] = model_training(data_package[0], 2, '~')
        dev_data[text[:-1]] = data_package[1]
    '''Code for generating text from each of the models: uncomment below to generate 5 sentences from each model'''
    # for name, model in models.items():
    #     for i in range(5):
    #         print(name)
    #         print(generate(model))


    print('Results on dev set:')
    results = {}
    for name, data in dev_data.items():
        results[name] = 0
        for sentence in data:
            guess = ""
            scores = {}
            for name2, model in models.items():
                perplex = use_model(model, sentence)
                scores[name2] = perplex
            temp = min(scores.values())
            guess = [key for key in scores if scores[key] == temp]
            if guess[0] == name:
                results[name] += 1
        results[name] /= len(data)
    print('Model Accuracy:')
    for key, value in results.items():
        print('{} {}\n'.format(key, value))

elif sys.argv[2] == '-test':
    f = open(sys.argv[1], "r")
    file_names = f.readlines()
    print("List of files we will be reading in: ", file_names)

    models = {}
    dev_data = {}
    for text in file_names:
        data_package = data_preprocessing(text, True)
        models[text[:-1]] = model_training(data_package[0], 2)
    test_data = test_data_processing(sys.argv[3])
    for sentence in test_data:
        guess = ""
        score = -1
        for name, model in models.items():
            perplex = use_model(model, sentence)
            if score == -1 and perplex != 'inf':
                score = perplex
                guess = name
            elif score > perplex:
                score = perplex
                guess = name
        print(guess[:-4])
else:
    print('Invalid input format')
