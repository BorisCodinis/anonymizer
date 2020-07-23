#import numpy as np
import json
#import sys
#from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.data import Sentence
from segtok.segmenter import split_single
#from sklearn.metrics import f1_score
from pprint import pprint
#import spacy
#from spacy.gold import GoldParse
#from spacy.scorer import Scorer
import re
#from spacy.lang.de import German
#import torch
#import flashtext
import time
#from transformers import AutoTokenizer, BertModel, AutoModelWithLMHead, \
#   BertForTokenClassification, AutoModelForTokenClassification, AutoConfig
from faker import Faker
from OSMPythonTools.overpass import Overpass

def load_model(name):
    if name == "spacy":
        return spacy.load("/content/drive/My Drive/Colab Notebooks/models/spacy")
        
    elif name == "flair":
        return SequenceTagger.load('models/de-ner-conll03-v0.4.pt')

    return 1
	
def split_sents(corpus):
    '''
    splits document string into multiple strings containing one sentence each
    corpus: corpus file, list of document strings
    returns list of lists of Sentence objects
    '''
    corp = []
    for doc in corpus['corpus']:
        sentences = [Sentence(sent, use_tokenizer = True) for sent in split_single(doc['text'])]
        corp.append(sentences)  
    return corp[:5]

def get_annotations(corpus, isPrediction):
    annotations = []
    if isPrediction:
        for doc in corpus:
            docAnnotations = []
            for sentence in doc:
                sentenceJson = sentence.to_dict('ner')
                for entity in sentenceJson['entities']:
                    docAnnotations.append((entity['labels'][0].value, entity['start_pos'], entity['end_pos']))
            annotations.append(docAnnotations)
        return annotations
    else:
        for doc in corpus:
            docAnnotations = []
            for entity in doc['annotations']:
                docAnnotations.append((entity['label'], entity['start_offset'], entity['end_offset']))
            annotations.append(docAnnotations)
        return annotations



def predict_corpus(corpus):
    '''
    predicts each sentence of the corpus separately
    corpus: List of lists of sentence Objectss
    returns same format but with prediction values; also prints out avg time 
        spent predicting per document
    '''
    model = load_model('flair') 
    
    times = [] 
    predicted_corpus = []
    for doc in corpus:
        start = time.time()
        model.predict(doc)
        times.append(time.time() - start)
        predicted_corpus.append(doc)
    print("Avg prediction time: " + str(sum(times)/len(times)))
    return predicted_corpus

def substitute_names(docString, nameString):
    '''
    Takes a whole document and the corrosponding name string found
    doc: string with the whole document
    nameString: name to be changed
    returns: docString with fake name
    '''

    faker = Faker(['de_DE', 'bs_BA', 'cs_CZ', 'dk_DK', 'en_GB', 'en_AU', 
                  'en_CA', 'es_ES', 'et_EE', 'fi_FI', 'fr_FR', 'hr_HR',
                  'it_IT', 'hu_HU', 'lt_LT', 'nl_NL', 'no_NO', 'pl_PL',
                  'pt_BR', 'sv_SE', 'tr_TR'])
    for name in nameString.split(' '):
        docString = re.subn(name, faker.first_name(), docString)[0] 
    return docString

def get_city(locString):
    '''
    tries to get the city the locString is located in with openStreetMap API
    locString: string of location found by NER-Tagger
    returns: 'city' field of osm-overpass query
    '''

    

def anonymize(corpus):
    '''
    corpus (input): list of lists of sentences per doc to be anonymized.  
    sent (flag): (default) True = list entries are sentences; False = list
        entries are documents
    returns list of anonymized documents
    '''

    
    corpus_text = split_sents(corpus)
    corpus_annot = get_annotations([doc for doc in corpus['corpus']], 0) #start_offset in datei umbenennen

    predicted_corpus = predict_corpus(corpus_text)

    predicted_annot = get_annotations(predicted_corpus, 1)

    pprint(predicted_annot)
    pprint(corpus_annot)
    exit()
    anonymized_docs = []
    startTime = time.time()
    for doc in predicted_corpus:
               
        
        entsWithLabel = []
        seperator = " "
        docString = seperator.join([sen.to_original_text() for sen in doc])
        for sentence in doc:
            sentenceJson = sentence.to_dict('ner')
            for entity in sentenceJson['entities']:
                
                label = entity['labels'][0].value
                entsWithLabel.append((label, entity['text']))
        for entity in entsWithLabel:
            label = entity[0]
            if label == "PER":
                
                name = entity[1]
                docString = substitute_names(docString, name)
            #elif label == "LOC":
            #    location = entity[1]
            #    get_city()
        anonymized_docs.append(docString)
        pprint(anonymized_docs)
    print('time needed for loop hell: ' + str(time.time() - startTime))
