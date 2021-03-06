from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
res = nlp.annotate("You are the best one! I am feeling hungry. I can run the code. Neighbours hate him. She is too dumb and shy",
                   properties={
                       'annotators': 'parse,sentiment,lemma',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
for s in res["sentences"]:
    print ("'%s': \t %s %s" % (
        " ".join([t["word"] for t in s["tokens"]]),
        s["sentimentValue"], s["sentiment"]))

