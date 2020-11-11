import numpy as np
# import nltk
import math
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.corpus import wordnet
import pickle
import spacy
import os
import sys
from pysbd.utils import PySBDFactory

nlp = spacy.load("en_core_web_lg")  # make sure to use larger model!

# nlp2 = spacy.load("en_core_web_lg")
# nlp2.add_pipe(PySBDFactory(nlp2), before="parser")

# nlp = spacy.load("en_core_web_md")  # make sure to use larger model!
# nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!

# input_file_name=sys.argv[1]
input_file_name = 'test_input.txt'

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
    qa_list=[]
    with open(fname, 'r') as fp:
        for line in fp:
            l = line.strip('\n').split(':')
            if len(l) > 0:
                if 'QuestionID' in line:
                    q_id = l[1].strip()
                    qa_list.append(q_id)
                elif 'Question:' in line:
                    q = l[1]
                elif 'Answer' in line:
                    a = l[1].split(' | ')
                # Assumes 'Difficulty' denotes end of qid
                elif 'Difficulty' in line:
                    d =l[1]
                    qa[q_id] = {'Question': q, 'Answer': a, 'Difficulty': d}
    return qa, qa_list


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
    end=t_idx+k
    if(t_idx-k<0):
        start=0
    if(t_idx+k>len(text)):
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
def get_best_context_w_weight(story_data, question_data, orig_story, attribute_dict, k, q_type, weight_dict, bump_word, q_words):
    best_context = get_context_words_span(story_data, k, 0)
    best_context_weight = 0
    for t_idx in range(len(story_data)):
        context_words = get_context_words_span(story_data, k, t_idx)
        curr_context_weight = 0
        #word level comparisons
        for q_word in question_data:
            if q_word[0].text.lower() in q_words:
                continue
            for s_word in context_words:
                for w_type in weight_dict: 
                    if(w_type == 'TEXT'):
                        if q_word[0].text == bump_word and q_word[0].similarity(s_word[0]) > .75:
                            b_weight = weight_dict["BUMP"]
                        else:
                            b_weight = 1
                        curr_context_weight += q_word[0].similarity(s_word[0]) * weight_dict[w_type] * b_weight
                    elif(w_type == 'POS'):
                        curr_attr = attribute_dict[q_type][w_type]
                        if(s_word[0].pos_ in curr_attr):
                            curr_context_weight += curr_attr[s_word[0].pos_] * weight_dict[w_type]
                    elif(w_type =='ENT'):
                        continue
                    else:
                        continue

        #context level comparisons
        if('ENT' in weight_dict):
            entities = []
            # Check if current context contains any entities
            tokenized_s = [token for token in orig_story]
            context_start_idx = tokenized_s[context_words[0][1]].idx
            context_end_idx = tokenized_s[context_words[-1][1]].idx + len(tokenized_s[context_words[-1][1]].text) - 1
            for ent in orig_story.ents:
                start_idx = ent.start_char
                end_idx = ent.end_char
                if start_idx >= context_start_idx and end_idx <= context_end_idx:
                    entities.append(ent)
                elif start_idx > context_end_idx:
                    break

            # if q_type in attribute_dict:
            #     curr_attr = attribute_dict[q_type]["ENT"]
            #     # print(curr_attr, file=sys.stderr)
            # else:
            #     curr_attr = attribute_dict["Generic"]["ENT"]

            curr_attr = attribute_dict[q_type]['ENT']
            for ent in entities:
                if ent.label_ in curr_attr:
                    # curr_context_weight += curr_attr[ent.label_] * weight_dict['ENT']
                    curr_context_weight += curr_attr[ent.label_] * weight_dict['ENT'] * 4
        if curr_context_weight > best_context_weight:
            best_context_weight = curr_context_weight
            best_context = context_words
            best_ents = entities
    return best_context, best_ents


# Removes any words with POS in filter tags from text, returns filtered text
def filter_by_POS(tagged_text, filter_tags):
    pos_text = [tagged[0].pos_ for tagged in tagged_text]
    filter_idxs = [idx for idx in range(len(pos_text)) if pos_text[idx] in filter_tags]
    filtered_text = [w for i, w in enumerate(tagged_text) if i not in filter_idxs]
    return filtered_text


def filter_by_stopwords(text, stopwords):
    filtered_text = [w for w in text if w[0].text not in stopwords]
    return filtered_text


def vectorize_list(text):
    str1 = ""
    for ele in text:  
        str1 += ele+" "
    vectorized = nlp(str1)
    return vectorized


# finds what two word question phrases appear in our data and the number of times they occur
# returns dict of form {two word phrase: count, ...}
def get_q_words_count(nlp_q, nlp_a):
    tokenized_q = [token.text for token in nlp_q]
    tokenized_a = [[token.text for token in a] for a in nlp_a]
    a_lens = [len(a) for a in tokenized_a]
    for i, w in enumerate(tokenized_q):
        if w.lower() in q_words:
            increase_sim_weight = False
            # I started going through the question types one-by one and recognizing some special cases
            # Here I outline those cases, while also clustering the question types into broader categories
            if w.lower() == 'what':
                if nlp_q[i + 1].pos_ == 'NOUN':
                    # Extra weight/more emphasis should be placed on this NOUN
                    # ex) what color?  green.
                    increase_sim_weight = True
                elif nlp_q[i + 1].pos_ == 'VERB':
                    # Extra weight/more emphasis should be placed on this VERB
                    # ex) what walks?  animals walk.
                    increase_sim_weight = True
                q2 = w.lower() + ' ' + nlp_q[i + 1].text
            elif w.lower() == 'who':
                if nlp_q[i + 1].pos_ == 'VERB':
                    # Extra weight/more emphasis should be placed on this VERB
                    # Who ran?  John ran.
                    increase_sim_weight = True
                elif nlp_q[i + 1].text == 'is':
                    # Do something special here?
                    pass
                q2 = w.lower() + ' ' + nlp_q[i + 1].text
            elif w.lower() == 'which':
                if nlp_q[i + 1].pos_ == 'NOUN':
                    # Extra weight/more emphasis should be placed on this NOUN
                    # ex) which day?  Monday.
                    increase_sim_weight = True
                q2 = w.lower() + ' ' + nlp_q[i + 1].text
            elif w.lower() == 'whose':
                if nlp_q[i + 1].pos_ == 'NOUN':
                    # Extra weight/more emphasis should be placed on this NOUN
                    # ex) whose house?  John's house.
                    increase_sim_weight = True
                q2 = w.lower() + ' ' + nlp_q[i + 1].text
            # I went through each of these, I don't think anything needs to be done
            # Keeping them like this for the time being in case we want to add any special rules
            elif w.lower() == 'how':
                q2 = w.lower() + ' ' + nlp_q[i + 1].text
            elif w.lower() == 'when':
                q2 = w.lower() + ' ' + nlp_q[i + 1].text
            elif w.lower() == 'where':
                q2 = w.lower() + ' ' + nlp_q[i + 1].text
            elif w.lower() == 'why':
                q2 = w.lower() + ' ' + nlp_q[i + 1].text
                # print(w.lower() + ' ' + nlp_q[i + 1].text + ' (' + nlp_q[i + 1].pos_ + ')')

            else:
                # q2 = w.lower() + ' ' + tokenized_q[i + 1]
                q2 = w.lower() + ' ' + nlp_q[i + 1].text

            if q2 not in list(q_2word_counts.keys()):
                # q_2word_counts[q2] = 1
                q_2word_counts[q2] = {}
                q_2word_counts[q2]['Count'] = 1
                q_2word_counts[q2]['ENT'] = {}
                # q_2word_counts[q2]['NP']= []
                q_2word_counts[q2]['POS'] = {}
                q_2word_counts[q2]['Avg Ans Len'] = a_lens  # Count lengths for now, take average at the end
                if increase_sim_weight:
                    q_2word_counts[q2]['Inc Sim Weight'] = True  # increase the similarity weight of the 2nd q word
                else:
                    q_2word_counts[q2]['Inc Sim Weight'] = False
            else:
                # q_2word_counts[q2] += 1
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


def get_q_type(question, q_words):
    tokenized_q = [token.text for token in question]
    bump_word = ""
    for i, token in enumerate(tokenized_q):
        if token.lower() in q_words:
            q_type = token.lower() + ' ' + question[i + 1].text
            if q_type not in list(q_2word_counts.keys()):
                q_type = token.lower() + ' ' + question[i + 1].pos_
                if q_type not in list(q_2word_counts.keys()):
                    q_type = 'Generic'
            if q_2word_counts[q_type]['Inc Sim Weight']:
                bump_word = question[i + 1].text
            return q_type, bump_word
    # print('stop')
    return "Generic", bump_word

# ===========================
# ===========================


#######Load Data####### these are test sets
stories = {}
questions = {}
for fname in os.listdir(os.getcwd() + '/data'):
    id = fname.split('.')[0]
    story_data = load_story('data/' + id + '.story')
    question_data, _ = load_QA('data/' + id + '.answers')
    stories[id] = story_data
    questions[id] = question_data

# Test set 1
for fname in os.listdir(os.getcwd() + '/testset1'):
    id = fname.split('.')[0]
    story_data = load_story('testset1/' + id + '.story')
    question_data, _ = load_QA('testset1/' + id + '.answers')
    stories[id] = story_data
    questions[id] = question_data

for fname in os.listdir(os.getcwd() + '/extra-data'):
    if '.answers' in fname:
        id = fname.split('.answers')[0]
        question_data, _ = load_QA('extra-data/' + id + '.answers')
        questions[id] = question_data
    else:
        id = fname.split('.story')[0]
        story_data = load_story('extra-data/' + id + '.story')
        stories[id] = story_data

#######yper parameters#######
# k = 5
default_weights = {"TEXT": 2.2, "POS": 1.06, "ENT": 4.13, "BUMP": 3.81, 'K': 3}
# default_k=4
# bump_weight = 2  # == 1 does nothing, should be greater than 1
# q_words = ['who', 'what', 'when', 'where', 'why', 'how', 'whose', 'which', 'did', 'are']  # couple weird ones here
####################################
filter_pos_tags = ['PUNCT', 'DET', 'SPACE']
# filter_pos_tags = ['PUNCT', 'DET', 'SPACE', 'ADV', 'AUX', 'PRON', 'ADP']

stop_words = nlp.Defaults.stop_words
# q_words = ['who', 'what', 'when']
q_words = ['who', 'what', 'when', 'where', 'why', 'how', 'whose', 'which']

####################################
#
# ######Build Dictionary for Question Types#######
# q_2word_counts = {}  # attribute dictionary
# id_to_type = {}  # link q to type
# for story_id in list(questions.keys()):
#     story_qa = questions[story_id]
#     for question_id in list(story_qa.keys()):
#         question = story_qa[question_id]['Question']
#         answers = story_qa[question_id]['Answer']
#         tokenized_q = [token.text for token in nlp(question)]
#         nlp_a = [nlp(a) for a in answers]
#         q_type = get_q_words_count(nlp(question), nlp_a)
#         id_to_type[question_id] = q_type  # TODO: In theory, this is just a training step.  Thus, id_to_type needs to be removed, since it is referenced in ##run##
# # q_2word_counts = {k: v for k, v in sorted(q_2word_counts.items(), key=lambda item: item[1], reverse=True)}
#
#
# new_q2 = {}
# for k1 in q_2word_counts.keys():
#     # if q_2word_counts[k1] < 10:
#     #     new_key = [token for token in nlp(k1)]
#     #     new_key = new_key[0].text + ' ' + new_key[1].pos_
#     #     if new_key not in new_q2:
#     #         new_q2[new_key] = q_2word_counts[k1]
#     #     else:
#     #         new_q2[new_key] += q_2word_counts[k1]
#     # else:
#     #     new_q2[k1] = q_2word_counts[k1]
#     if q_2word_counts[k1]['Count'] < 10:
#         new_key = [token for token in nlp(k1)]
#         new_key = new_key[0].text + ' ' + new_key[1].pos_
#         if new_key not in new_q2:
#             new_q2[new_key] = q_2word_counts[k1]
#         else:
#             for k2 in q_2word_counts[k1].keys():
#                 if k2 == 'POS' or k2 == 'ENT':
#                     for k3 in q_2word_counts[k1][k2].keys():
#                         if k3 not in new_q2[new_key][k2].keys():
#                             new_q2[new_key][k2][k3] = q_2word_counts[k1][k2][k3]
#                         else:
#                             new_q2[new_key][k2][k3] += q_2word_counts[k1][k2][k3]
#                 elif k2 == "Inc Sim Weight":
#                     new_q2[new_key][k2] = q_2word_counts[k1][k2]
#                 else:
#                     new_q2[new_key][k2] += q_2word_counts[k1][k2]
#     else:
#         new_q2[k1] = q_2word_counts[k1]
# q_2word_counts = new_q2
# # q_2word_counts = {k: v for k, v in sorted(q_2word_counts.items(), key=lambda item: item[1], reverse=True)}
# # np.save('sorted_qtypes', q_2word_counts)
# get_avg_ans_len()
#
# # Normalize q_2word_counts values
# norm_keys = ['ENT', 'POS']  # values to normalize
# for q2 in q_2word_counts.keys():
#     for k in norm_keys:
#         count = 0
#         for item in q_2word_counts[q2][k].keys():
#             count += q_2word_counts[q2][k][item]
#         for item in q_2word_counts[q2][k].keys():
#             q_2word_counts[q2][k][item] = q_2word_counts[q2][k][item] / count

# f = open('attribute_dictionary', 'wb')
# pickle.dump(q_2word_counts, f)
# f.close()

# #######LOAD INPUT FOR TESTING #################
q_2word_counts=np.load('./attribute_dictionary_MASTER', allow_pickle=True)
loaded_weights=np.load('./tuned_weights_all', allow_pickle=True)
count = 0

# Add a 'Generic' feature to our q_2word_counts, a weighted avg of all other features
# This is in case we come across a question we've never seen
for k in q_2word_counts.keys():
    count += q_2word_counts[k]['Count']
nkeys = len(list(q_2word_counts.keys()))
gen_keys = ['ENT', 'POS']
generic_count = {'ENT': {}, 'POS': {}, 'Avg Ans Len': 5, 'Inc Sim Weight': False}
for k1 in q_2word_counts.keys():
    cw = q_2word_counts[k1]['Count'] / count
    for k2 in gen_keys:
        for k3 in q_2word_counts[k1][k2].keys():
            if k3 not in generic_count[k2]:
                generic_count[k2][k3] = q_2word_counts[k1][k2][k3] * cw
            else:
                generic_count[k2][k3] += q_2word_counts[k1][k2][k3] * cw
q_2word_counts['Generic'] = generic_count



test_stories={}
test_questions={}
#  id = fname.split('.answers')[0]
#         question_data = load_QA('extra-data/' + id + '.answers')
#         questions[id] = question_data
fn=open(input_file_name)
lnum=0
wdir=""
ordered_ids=[]
ordered_qs={}
for l in fn:
    if lnum==0:
        wdir=l.strip('\n')
        lnum+=1
        continue
    
    id=l.strip('\n')
    fts=(wdir+id+".story")
    story_data=load_story(fts)
    test_stories[id]=story_data
    fta=(wdir+id+".answers")
    q_data, lst=load_QA(fta)
    test_questions[id]=q_data
    ordered_ids.append(id)
    ordered_qs[id]=lst

fn.close()
outputs=[]

######run#######
for story_id in ordered_ids:
    story_qa = test_questions[story_id]
    story = test_stories[story_id]['TEXT']

    ####NEW####
    # story_sents = list(nlp2(story).sents)
    # test1 = [token for token in nlp(story_sents[0].text)]
    # test2 = [token for token in nlp(story)]
    tagged_text = [[token, i] for i, token in enumerate(nlp(story))]
    filtered_s = filter_by_POS(tagged_text, filter_pos_tags)
    filtered_s = filter_by_stopwords(filtered_s, stop_words)
    # vectorized_s = vectorize_list(filtered_s_text)

    for question_id in ordered_qs[story_id]:
        question = story_qa[question_id]['Question']
        answer = story_qa[question_id]['Answer']
        # q_type = id_to_type[question_id]  # TO DO: This takes information from a pre-processing step, thus should be removed
        q_type, bump_word = get_q_type(nlp(question), q_words)
        tagged_q = [[token, i] for i, token in enumerate(nlp(question))]
        filtered_q = filter_by_POS(tagged_q, filter_pos_tags)
        filtered_q = filter_by_stopwords(filtered_q, stop_words)
        # vectorized_q = vectorize_list(filtered_q_text)

        # k = math.ceil(q_2word_counts[q_type]['Avg Ans Len'] / 2)
        k = 5

        q_type2 = q_type.split()
        if q_type2[1].islower():
            tmp = [token for token in nlp(q_type)]
            q_type2 = tmp[0].text + ' ' + tmp[1].pos_
        else:
            q_type2 = q_type2[0] + ' ' + q_type2[1]

        used_weights = default_weights
        if q_type2 in loaded_weights.keys():
            # if loaded_weights[q_type2]["TEXT"] > 0:
            #     used_weights = loaded_weights[q_type2]
            used_weights = loaded_weights[q_type2]

        best_context, ents = get_best_context_w_weight(filtered_s, filtered_q, nlp(story), q_2word_counts, k, q_type, used_weights, bump_word, q_words)
        
        # print(question,file=sys.stderr)
        # print(story_qa[question_id]['Answer'],file=sys.stderr)
        # print(best_context,file=sys.stderr)
        # print('\n',file=sys.stderr)
        print('QuestionID: ', question)
        print('Answer: ', answer)
        ans = ''
        for w in best_context:
            ans += w[0].text + ' '
        print('Answer: ' + ans)
        print('Entities in answer: ', [e.label_ for e in ents], '\n')
#         outputs.append([question_id, best_context])


# for output in outputs:
#     print('QuestionID: '+output[0])
#     print('Answer: ' + output[1] + "\n")


#TODO: write function for getting entire context not filtered
#function for output
#cleanup k issue
#check with scoring mechanism
#remove stop dep for nltk
#remove wordnet dep
#write install script
#write README