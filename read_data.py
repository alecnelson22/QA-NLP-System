import numpy as np
import nltk
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

import spacy
import os


nlp = spacy.load("en_core_web_md")  # make sure to use larger model!
# nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!

# # Load a story into separate sentences
# # TODO: some quality control required (see below)
# def load_story_sentences(fname):
#     read_text = False
#     sentences = []
#     sentence = ''
#     with open(fname, 'r') as fp:
#         for line in fp:
#             d = line.split(':')
#             if read_text:
#                 # Check if current line contains end of sentence
#                 if '.\n' in line:
#                     sentence += line.split('.\n')[0]
#                     sentences.append(sentence)
#                     sentence = ''
#                 # Assumes period is followed by single spaces
#                 # TODO: This would count an abbreviation as a sentence end
#                 # TODO: items like 'data/1999-W02-5.story' line 40 needs to be handled
#                 elif '. ' in line:
#                     s = line.split('. ')
#                     for i in range(len(s)):
#                         sentence += s[i]
#                         if i < len(s) - 1:
#                             sentences.append(sentence)
#                             sentence = ''
#                 else:
#                     sentence += line.strip('\n') + ' '
#             elif 'HEADLINE' in line:
#                 headline = d[1].strip('\n')
#             elif 'DATE' in line:
#                 date = d[1].strip('\n')
#             elif 'STORYID' in line:
#                 id = d[1].strip('\n')
#             elif 'TEXT' in line:
#                 read_text = True
#     return {id: {'HEADLINE': headline, 'DATE': date, 'TEXT': sentences}}


# Load story as single string, returns data in dict
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
    # return {id: {'HEADLINE': headline, 'DATE': date, 'TEXT': story}}
    return {'HEADLINE': headline, 'DATE': date, 'TEXT': story}


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


# Grabs words k to the left and k to the right of word at target index t_idx
def get_context_words(text, k, t_idx, split=False):
    if split:
        text = text.split()
    if k == 0:
        l_words = text[0:t_idx]
        r_words = text[t_idx+1:]
        words = l_words + r_words
    else:
        if t_idx-k > 0:
            start = t_idx - k
        else:
            start = 0
        l_words = text[start:t_idx]
        if t_idx+k < len(text):
            end = t_idx+k
        else:
            end = len(text)
        r_words = text[t_idx:end]
        words = l_words + r_words
    return words


def get_context_words_span(text, k, t_idx):
    start=t_idx-k
    end=t_idx+k+1
    if(t_idx-k<0):
        start=0
    if(t_idx+k+1>len(text)):
        end=len(text)

    words= text[start:end]
    # print(words)
    return(words)


# Finds and returns the context with size k from a story text that
# contains the highest number of words from a question text
def get_best_context(story, question, k):
    best_context_matches = 0
    for t_idx in range(len(story)):
        context_words = get_context_words(story, k, t_idx)
        curr_context_matches = 0
        # for q_word in question.split():
        for q_word in question:
            # TODO: if duplicates exist, this still only counts 1
            # Could add a custom scoring function here
            if q_word in context_words:
                curr_context_matches += 1
        if curr_context_matches > best_context_matches:
            best_context_matches = curr_context_matches
            best_context = context_words
    return best_context


def get_best_context_w_pos(story, story_pos, question, question_pos,k):
    best_context_matches = 0
    for t_idx in range(len(story)):
        context_words = get_context_words(story, k, t_idx,False)
        context_pos = get_context_words(story_pos, k, t_idx,False)
        # print(context_words,context_pos)
        # asdf
        curr_context_matches = 0
        for q_word,q_pos in zip(question, question_pos):
            # print(q_word,q_pos)
            # TODO: if duplicates exist, this still only counts 1
            for cword, cpos in zip(context_words,context_pos):
                if q_word == cword and cpos==q_pos:
                    curr_context_matches += 1
                    # print('yay')
        if curr_context_matches > best_context_matches:
            best_context_matches = curr_context_matches
            best_context = context_words
    return best_context


#input: story & q are spacy objs. q_type is the question type, weight dict defines how to distubute weights among attributes
def get_best_context_w_weight(story, question, attribute_dict, k, q_type, weight_dict):
    best_context_weight = 0
    for t_idx in range(len(story)):
        context_words = get_context_words_span(story, k, t_idx)  # context_words is a spacy doc
        curr_context_weight = 0
        #word level comparisons
        for q_word in question:
            for s_word in context_words:
                for w_type in weight_dict: 
                    if(w_type == 'TEXT'):
                        curr_context_weight += q_word.similarity(s_word) * weight_dict[w_type]
                    elif(w_type == 'POS'):
                        curr_attr = attribute_dict[q_type][w_type]
                        if(s_word.pos_ in curr_attr):
                            curr_context_weight += curr_attr[s_word.pos_] * weight_dict[w_type]
                    elif(w_type =='ENT'):
                        continue
                    else:
                        print("Attribute Type currently not supported")

        #context level comparisons
        if('ENT' in weight_dict):
            entities = [ent.label_ for ent in context_words.ents]
            curr_attr = attribute_dict[q_type]["ENT"]
            for ent in entities:
                if ent in curr_attr:
                   curr_context_weight += curr_attr[ent] * weight_dict[w_type]

        # print(curr_context_weight)
        if curr_context_weight > best_context_weight:
            best_context_weight = curr_context_weight
            best_context = context_words
    print(best_context_weight)
    return best_context


# Removes any words with POS in filter tags from text, returns filtered text
def filter_by_POS(text, filter_tags):
    # tagged_text = get_POS_tags(text)
    tagged_text = [[token.text, token.pos_] for token in text]
    pos_text = [tagged[1] for tagged in tagged_text]
    word_text = [tagged[0] for tagged in tagged_text]
    filter_idxs = [idx for idx in range(len(pos_text)) if pos_text[idx] in filter_tags]
    filtered_text = [w for i, w in enumerate(word_text) if i not in filter_idxs]
    filtered_pos=[w for i, w in enumerate(pos_text) if i not in filter_idxs]
    return filtered_text, filtered_pos


def filter_by_stopwords(text, stopwords):
    filtered_text = [w for w in text if w not in stopwords]
    return filtered_text


def vectorize_list(text):
    str1 = ""
    for ele in text:  
        str1 += ele+" "
    vectorized = nlp(str1)
    return vectorized


# finds what two word question phrases appear in our data and the number of times they occur
# returns dict of form {two word phrase: count, ...}
def get_q_words_count(tokenized_q, nlp_a):
    tokenized_a = [[token.text for token in a] for a in nlp_a]
    a_lens = [len(a) for a in tokenized_a]
    # a_avg_len = math.ceil(np.mean(a_lens))
    for i, w in enumerate(tokenized_q):
        if w.lower() in q_words:
            q2 = w.lower() + ' ' + tokenized_q[i + 1]
            if q2 not in list(q_2word_counts.keys()):
                q_2word_counts[q2] = {}
                q_2word_counts[q2]['Count'] = 1
                q_2word_counts[q2]['ENT'] = {}
                # q_2word_counts[q2]['NP']= [] 
                q_2word_counts[q2]['POS'] = {}
                q_2word_counts[q2]['Avg Ans Len'] = a_lens  # Count lengths for now, take average at the end
            else:
                q_2word_counts[q2]['Count'] += 1
                q_2word_counts[q2]['Avg Ans Len'].extend(a_lens)

            for a in nlp_a:
                for ent in a.ents:
                    if ent.label_ not in q_2word_counts[q2]['ENT']:
                        q_2word_counts[q2]['ENT'][ent.label_] = 1
                    else:
                        q_2word_counts[q2]['ENT'][ent.label_] += 1
                for token in a:
                    if (not token.is_stop and not token.pos_ == 'SPACE'):
                        tag = token.pos_
                        if tag not in q_2word_counts[q2]['POS']:
                            q_2word_counts[q2]['POS'][tag] = 1
                        else:
                            q_2word_counts[q2]['POS'][tag] += 1
    return q2


def get_avg_ans_len():
    for q_type in q_2word_counts.keys():
        q_2word_counts[q_type]['Avg Ans Len'] = math.ceil(np.mean(q_2word_counts[q_type]['Avg Ans Len']))

# ===========================
# ===========================


#######Load Data#######
stories = {}
questions = {}
for fname in os.listdir(os.getcwd() + '/data'):
        id = fname.split('.')[0]
        story_data = load_story('data/' + id + '.story')
        question_data = load_QA('data/' + id + '.answers')
        stories[id] = story_data
        questions[id] = question_data


#######Build Dictionary for Question Types#######
q_words = ['who', 'what', 'when', 'where', 'why', 'how', 'whose', 'which']
q_2word_counts = {}  # attribute dictionary
id_to_type = {}  # link q to type
for story_id in list(questions.keys()):
    story_qa = questions[story_id]
    for question_id in list(story_qa.keys()):
        question = story_qa[question_id]['Question']
        answers = story_qa[question_id]['Answer']
        tokenized_q = [token.text for token in nlp(question)]
        nlp_a = [nlp(a) for a in answers]
        q_type = get_q_words_count(tokenized_q, nlp_a)
        id_to_type[question_id] = q_type
get_avg_ans_len()


# (TODO: hyperparameter k should be based on average answer length for that given question type?)
#######yper parameters#######
k = 5
weights = {"TEXT": .5, "POS": .1, "ENT": 1}
filter_pos_tags = ['PUNCT', 'DET', 'SPACE', 'ADV', 'AUX', 'PRON', 'ADP']
stop_words = set(stopwords.words('english'))
####################################


######run#######
for story_id in list(questions.keys()):
    story_qa = questions[story_id]
    story = stories[story_id]['TEXT']

    for question_id in list(story_qa.keys()):
        question = story_qa[question_id]['Question']
        answer = story_qa[question_id]['Answer']  # this is a list
        q_type = id_to_type[question_id]

        filtered_q, filtered_q_pos = filter_by_POS(nlp(question), filter_pos_tags)
        filtered_s_text, filtered_s_pos = filter_by_POS(nlp(story), filter_pos_tags)

        filtered_q = filter_by_stopwords(filtered_q, stop_words)
        filtered_s = filter_by_stopwords(filtered_s_text, stop_words)

        # # Build synonym list for words in question
        # # TODO: clean this list
        # synonyms = []
        # for word in filtered_q:
        #     for synset in wordnet.synsets(word):
        #         for lemma in synset.lemmas():
        #             synonyms.append(lemma.name())
        # filtered_q.extend(synonyms)

        vectorized_q = vectorize_list(filtered_q)
        vectorized_s = vectorize_list(filtered_s)

        k = math.ceil(q_2word_counts[q_type]['Avg Ans Len'] / 2)

        best_context = get_best_context_w_weight(vectorized_s, vectorized_q, q_2word_counts, k, q_type, weights)
        
        print(question)
        print(story_qa[question_id]['Answer'])
        print(best_context)
        print('\n')




#TODO: write function for getting entire context not filtered
#function for output
#cleanup k issue
#check with scoring mechanism
#remove stop dep for nltk
#remove wordnet dep
#write install script
#write README