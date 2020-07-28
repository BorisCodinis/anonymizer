import json
from math import sqrt, pi, exp
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single
from pprint import pprint
import re
import time
from faker import Faker
from OSMPythonTools.overpass import Overpass

DOC_COUNT = 200

def load_model(name):
    if name == "spacy":
        return spacy.load("/content/drive/My Drive/Colab Notebooks/models/spacy")
        
    elif name == "flair":
        return SequenceTagger.load('models/multi-ner.pt')

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

def create_distribution(prec, entsPerDoc):
    def prop_for(k):   
        expect = prec * entsPerDoc
        var = prec * entsPerDoc * (1-prec)
        k1 = pow(k - expect, 2)
        k2 = -k1/2 * pow(var, 2)
        return (1./(sqrt(2. * pi) * var)) * exp(k2)

    return prop_for


def get_entity_list(predictedCorpus, corpus):
    perCount = 0
    locCount = 0
    predictedPers = []
    predictedLocs = []
    annotatedPers = []
    annotatedLocs = []
    seperator = " "

    for doc in predictedCorpus: 
        for sentence in doc: #one document has multiple Sentence objects,
            sentenceJson = sentence.to_dict('ner')
            for entity in sentenceJson['entities']:
                label = entity['labels'][0].value 
                if label == 'PER':
                    predictedPers.append(entity['text'])
                elif label == 'LOC':
                    predictedLocs.append(entity['text'])
    
    for doc in corpus: # Iterate over annotated corpus
        tmpPerString = []
        tmpLocString = []

        for entity in doc['annotations']:
            entText = doc['text'][entity['start_offset']:entity['end_offset']] \
                .replace('\r\n', '').replace('\n', '').strip()
            label = entity['label'] 
                
            if (type(label) == int and label == 20):
                entText = re.sub(' +', ' ', entText)
                annotatedPers.append(entText.strip())
                perCount += 1

            if (type(label) == str and label[-3:] == 'PER'):
                if label[0] == 'B': 
                    perCount += 1
                    if len(tmpPerString) == 0:
                        tmpPerString.append(entText.strip())
                    
                    else: 
                        annotatedPers.append(seperator.join(tmpPerString))
                        tmpPerString.clear()
                        tmpPerString.append(entText.strip())
                else:
                    tmpPerString.append(entText.strip())




            if (type(label) == int and label == 23):
                entText = re.sub(' +', ' ', entText)
                annotatedLocs.append(entText.strip())
                locCount += 1

            if (type(label) == str and label[-3:] == 'LOC'):
                if label[0] == 'B': 
                    locCount += 1
                    if len(tmpLocString) == 0:
                        tmpLocString.append(entText.strip())
                    
                    else: 
                        annotatedLocs.append(seperator.join(tmpLocString))
                        tmpLocString.clear()
                        tmpLocString.append(entText.strip())
                else:
                    tmpLocString.append(entText.strip())

        if len(tmpLocString) > 0:
            annotatedLocs.append(seperator.join(tmpLocString))



    
    return {'per': (predictedPers, annotatedPers, perCount/len(corpus)), 'loc': (predictedLocs, annotatedLocs, locCount/len(corpus))}

def print_ents(corpus):
    for doc in corpus:
        for i in doc['annotations']:
            label = i['label']
            if (type(label) == int and label == 20) or (type(label) == str and label[-3:] == 'PER'):
                pprint(doc['text'][i['start_offset']:i['end_offset']].replace("\r\n", "").strip())

def calc_results_for_label(predEnts, annotEnts, entsPerDoc, label = 'per'):
    tp = 0
    fp = 0
    #tn = 0
    fn = 0
    totalPerCount = len(annotEnts)
    for i in predEnts:
        if i in annotEnts:
            annotEnts.remove(i)
            tp += 1
        else:
            fp += 1
    fn = len(annotEnts)
    #tn = entCount - totalPerCount - fp
    return {'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'ppd': entsPerDoc}


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
    perResult = calc_results_for_label(entities['per'][0], entities['per'][1], entities['per'][2])
    locResult = calc_results_for_label(entities['loc'][0], entities['loc'][1], entities['loc'][2])
    pprint(perResult)
    pprint(locResult)
    calculator = create_distribution(perResult['recall'], perResult['ppd'])

    pprint((locResult['ppd']*locResult['recall']) + (perResult['ppd']*perResult['recall']))
    print(calculator(0))
    exit()
    anonymized_docs = []
    startTime = time.time()
    for doc in predictedCorpus:
               
        
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
