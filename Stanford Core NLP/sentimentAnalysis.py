from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
res = nlp.annotate("you are the best! I am feeling hungry. I can run the code. Neighbours hate him. She is too dumb and shy",
                   properties={
                       'annotators': 'parse,sentiment,lemma',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
for s in res["sentences"]:
    print ("%d: '%s': \t\t %s %s" % (
        s["index"],
        " ".join([t["word"] for t in s["tokens"]]),
        s["sentimentValue"], s["sentiment"]))
