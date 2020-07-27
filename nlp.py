#import numpy as np
import json
from math import sqrt
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

DOC_COUNT = 100

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
    for doc in corpus:
        sentences = [Sentence(sent, use_tokenizer = True) for sent in split_single(doc['text'])]
        corp.append(sentences)  
    return corp

def predict_corpus(corpus):
    '''
    predicts each sentence of the corpus separately
    corpus: List of lists of sentence Objectss
    returns same format but with prediction values; also prints out avg time 
        spent predicting per document
    '''
    model = load_model('flair') 
    
    times = [] 
    predictedCorpus = []
    predCount = 0
    c = 0
    for doc in corpus:
        start = time.time()
        model.predict(doc)
        predCount += 1
        timeNeeded = time.time() - start
        times.append(timeNeeded)
        predictedCorpus.append(doc)
        print(str(timeNeeded) + " Seconds nedded for predicting " + str(predCount) + ". document")
        c += 1
        if c == DOC_COUNT:
            break
    print("Total doc count: " + str(predCount))
    print("Avg prediction time: " + str(sum(times)/len(times)))
    return predictedCorpus


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

def get_entity_list(predictedCorpus, corpus):
    entCount = 0
    predictedPers = []
    annotatedPers = []
    seperator = " "
    for doc in predictedCorpus:
        
        totalEntsPerDoc = 0
        for sentence in doc:
            sentenceJson = sentence.to_dict('ner')
            for entity in sentenceJson['entities']:
                label = entity['labels'][0].value 
                if label == 'PER':
                    predictedPers.append(entity['text'])
    for doc in corpus:
        perCount = 0

        

        entCount += len(doc['annotations'])
        tmpString = []
        for entity in doc['annotations']:
            entText = doc['text'][entity['start_offset']:entity['end_offset']].replace('\r\n', '').replace('\n', '').strip()
            label = entity['label'] 
                
            if (type(label) == int and label == 20):
                entText = re.sub(' +', ' ', entText)
                annotatedPers.append(entText.strip())
            if (type(label) == str and label[-3:] == 'PER'):
                if label[0] == 'B': 
                    if len(tmpString) == 0:

                        tmpString.append(entText.strip())
                    else: 
                        annotatedPers.append(seperator.join(tmpString))
                        tmpString.clear()
                        tmpString.append(entText.strip())

                else:
                    tmpString.append(entText.strip())
        if len(tmpString) > 0:
            annotatedPers.append(seperator.join(tmpString))
    return (predictedPers, annotatedPers, entCount) 

def print_ents(corpus):
    for doc in corpus:
        for i in doc['annotations']:
            label = i['label']
            if (type(label) == int and label == 20) or (type(label) == str and label[-3:] == 'PER'):
                pprint(doc['text'][i['start_offset']:i['end_offset']].replace("\r\n", "").strip())

def calc_results_for_label(predEnts, annotEnts, entCount,label = 'per'):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    totalPerCount = len(annotEnts)
    for i in predEnts:
        if i in annotEnts:
            annotEnts.remove(i)
            tp += 1
        else:
            fp += 1
    fn = len(annotEnts)
    tn = entCount - totalPerCount - fp
    return {'precision': tp / (tp + fp), 'recall': tp / (tp + fn)}


def anonymize(corpus):
    '''
    corpus (input): list of lists of sentences per doc to be anonymized.  
    sent (flag): (default) True = list entries are sentences; False = list
        entries are documents
    returns list of anonymized documents
    '''

    corpusText = split_sents(corpus)
    predictedCorpus = predict_corpus(corpusText)
    entities = get_entity_list(predictedCorpus, corpus[:DOC_COUNT])
    calcResult = calc_results_for_label(entities[0], entities[1], entities[2])
    pprint(calcResult)
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
