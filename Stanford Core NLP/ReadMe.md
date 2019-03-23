Follow the steps to run the script:

1. Download Stanford CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/)
2. cd to directory "stanford-corenlp-full-2018-10-05"
3. Start server using: java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000
4. Run the script using: python sentimentAnalysis.py

output:

- You are the best one!: 	 4 Verypositive
- I am feeling hungry: 	   2 Neutral
- I can run the code: 	   2 Neutral
- Neighbours hate him: 	   1 Negative
- She is too dumb and shy: 1 Negative
