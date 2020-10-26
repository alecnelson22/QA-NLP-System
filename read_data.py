import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


# Load a story into separate sentences
def load_story_sentences(fname):
    read_text = False
    sentences = []
    sentence = ''
    with open(fname, 'r') as fp:
        for line in fp:
            d = line.split(':')
            if read_text:
                # Check if current line contains end of sentence
                if '.\n' in line:
                    sentence += line.split('.\n')[0]
                    sentences.append(sentence)
                    sentence = ''
                # Assumes period is followed by single spaces
                # TODO: This would count an abbreviation as a sentence end
                # TODO: items like 'data/1999-W02-5.story' line 40 needs to be handled
                elif '. ' in line:
                    s = line.split('. ')
                    for i in range(len(s)):
                        sentence += s[i]
                        if i < len(s) - 1:
                            sentences.append(sentence)
                            sentence = ''
                else:
                    sentence += line.strip('\n') + ' '
            elif 'HEADLINE' in line:
                headline = d[1].strip('\n')
            elif 'DATE' in line:
                date = d[1].strip('\n')
            elif 'STORYID' in line:
                id = d[1].strip('\n')
            elif 'TEXT' in line:
                read_text = True
    return {id: {'HEADLINE': headline, 'DATE': date, 'TEXT': sentences}}


# Load story as single string, adds to dict
def load_story(fname):
    read_text = False
    story = ''
    with open(fname, 'r') as fp:
        for line in fp:
            d = line.split(':')
            if read_text:
                if len(line.strip('\n')) > 0:
                    story += line.strip('\n') + ' '
            elif 'HEADLINE' in line:
                headline = d[1].strip('\n')
            elif 'DATE' in line:
                date = d[1].strip('\n')
            elif 'STORYID' in line:
                id = d[1].strip('\n')
            elif 'TEXT' in line:
                read_text = True
    return {id: {'HEADLINE': headline, 'DATE': date, 'TEXT': story}}


# Loads questions and answers from .answers files into dict
def load_QA(fname):
    qa = {}
    with open(fname, 'r') as fp:
        for line in fp:
            l = line.strip('\n').split(':')
            if len(l) > 0:
                if 'QuestionID' in line:
                    q_id = l[1]
                elif 'Question:' in line:
                    q = l[1]
                elif 'Answer' in line:
                    a = l[1].split(' | ')
                # Assumes 'Difficulty' denotes end of qid
                elif 'Difficulty' in line:
                    d =l[1]
                    qa[q_id] = {'Question': q, 'Answer': a, 'Difficulty': d}
    return qa


# Returns list of tuples (word, pos) given a string of text
def get_POS_tags(text):
    tokenized = nltk.word_tokenize(text)
    return nltk.pos_tag(tokenized)


# s = load_story_sentences('data/1999-W02-5.story')
s = load_story('data/1999-W02-5.story')

# Get POS tags for stories
for k in list(s.keys()):
    text = s[k]['TEXT']
    pos = get_POS_tags(text)

qa = load_QA('data/1999-w22-2.answers')
print('yer')