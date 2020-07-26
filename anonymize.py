import nlp
from pprint import pprint
import json
import sys

def load_corpus(filename):
    jsonFile = []
    with open(filename, "r") as f:
        for line in f:
            jsonFile.append(json.loads(line))
    return jsonFile

if __name__ == '__main__':
    corpus = load_corpus(sys.argv[1])
    #pprint(corpus)
    nlp.anonymize(corpus)
