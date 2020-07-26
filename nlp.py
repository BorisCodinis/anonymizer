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

DOC_COUNT = 5

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



def merge_annotations(predictedCorpus, corpus):
    def aggregate_entities(sentensizedCorpus):
        ents = []
        for doc in sentensizedCorpus:
            docEnts = []
            for sentence in doc:
                docEnts.append(sentence.to_dict('ner')['entities'])
            ents.append(docEnts)
        return ents
    predictedEnts = aggregate_entities(predictedCorpus)

    mergedAnnots = []
    if len(predictedEnts) != len(corpus):
        print("unmatching doc count; predicted docs: {}, inserted docs: {}".format(len(predictedEnts), len(corpus)))
        exit()
    for i in range(len(predictedEnts)):
        mergedAnnots.append((predictedEnts[i], corpus[i]['annotations']))
    return mergedAnnots

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
        #c += 1
        #if c == DOC_COUNT:
        #    break
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

def get_annotations(predictedCorpus, corpus):
    entCount = {'predicted': {'per': 0, 'loc': 0, 'total': 0}, 'annots': {'per': 0, 'loc': 0, 'total': 0}}
    
    annotations = []
    perPerDoc = []
    for doc in predictedCorpus:
        
        totalEntsPerDoc = 0
        #docAnnotations = []
        for sentence in doc:
            sentenceJson = sentence.to_dict('ner')
            totalEntsPerDoc += len(sentenceJson['entities'])
            for entity in sentenceJson['entities']:
                label = entity['labels'][0].value 
                if label == 'PER':
                    #print(entity['text'])
                    entCount['predicted']['per'] += 1
                if label == 'LOC':
                    #print(entity['text'])
                    entCount['predicted']['loc'] += 1
                #docAnnotations.append((label, entity['text']))
        entCount['predicted']['total'] += totalEntsPerDoc 
        #annotations.append(docAnnotations) 
        #perPerDoc.append(perCount)
    
    for doc in corpus:
        perCount = 0
        entCount['annots']['total'] += len(doc['annotations'])
        for entity in doc['annotations']:
            label = entity['label'] 
                
            if (type(label) == int and label == 20) or (type(label) == str and label[-3:] == 'PER'):
                entCount['annots']['per'] += 1
            if (type(label) == int and label == 23) or (type(label) == str and label[-3:] == 'LOC'):
                entCount['annots']['loc'] += 1
                
            #startPos = entity['start_offset']
            #endPos = entity['end_offset']
            #docAnnotations.append((entity['label'], doc['text'][startPos:endPos].replace("\r", "").replace("\n", "")))
        #annotations.append(docAnnotations)
        #perPerDoc.append(perCount)
    return entCount

def calc_results_for_label(predCount, annotCount, label):
    if annotCount[label] >= predCount[label]: #weniger gefunden als vorhanden
        tp = predCount[label]
        fp = tp * 0.15
        fn = annotCount[label] - tp
        tn = annotCount['total'] - annotCount[label] - fp
    else: #mehr gefunden als vorhanden
        tp = annotCount[label]
        fp = tp - predCount[label]
        fn = predCount[label] * 0.15 
        tn = annotCount['total'] - tp - fp

    #return {'precision': tp / (tp + fp), 'recall': tp / (tp + fn)}
    return ((tp*tn) - (fp*fn)) / sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))


def anonymize(corpus):
    '''
    corpus (input): list of lists of sentences per doc to be anonymized.  
    sent (flag): (default) True = list entries are sentences; False = list
        entries are documents
    returns list of anonymized documents
    '''

    
    corpusText = split_sents(corpus)
    #corpus_annot = get_annotations([doc for doc in corpus], 0)


    predictedCorpus = predict_corpus(corpusText)

    #predicted_annot = get_annotations(predicted_corpus, 1)

    #print(predicted_annot)
    #print(corpus_annot)
    #labelCountTuple = [(predicted_annot[i], corpus_annot[i]) for i in range(len(predicted_annot)) if corpus_annot[i]!=0 and predicted_corpus[i]!=0]
    #hasSameCount = [(lambda x, y: x == y)(tup[0] , tup[1]) for tup in labelCountTuple]
    #print(hasSameCount)
    #counter = 0
    #for i in hasSameCount:
    #    if i:
    #        counter += 1
    #pprint(merge_annotations(predicted_corpus, corpus[:DOC_COUNT]))
    #print(counter)
    entCounts = get_annotations(predictedCorpus, corpus)
    pprint(entCounts)
    calcResult = calc_results_for_label(entCounts['predicted'], entCounts['annots'], 'per')
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
