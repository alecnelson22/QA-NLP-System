# nlp-final
Final project for CS 6340 (Midpoint Release)
External Libraries Used:

spacy -- nlp library simlar to nltk. Used for creating word embeddings using pretrained model (install file will download it)
https://spacy.io/


Instructions after installation:

To run our program run 

python3 qy.py input.txt > output.txt


Time: Our preprocessing step takes around a minute. Each document in an input file should be processed in around 15s at the longest, depending on the 
number of questions


Both team members worked in close collaboration throught the process. Some specific tasks that can be attributed to an individual are:

Torin:
hyperparameter tuning set up, I/O, word embeddings

Alec:
I/O, constructing attribute dictionary containing statistical properties used in our computation, tuning our model

We don't have any known limitations for our system
