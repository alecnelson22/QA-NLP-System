import numpy as np
# import nltk
import math
# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.corpus import wordnet
import pickle
import spacy
import os
import sys
from pysbd.utils import PySBDFactory
import statistics
nlp = spacy.load("en_core_web_lg")  # make sure to use larger model!

nlp2 = spacy.load("en_core_web_lg")
nlp2.add_pipe(PySBDFactory(nlp2), before="parser")

# nlp = spacy.load("en_core_web_md")  # make sure to use larger model!
# nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!

if len(sys.argv)>1:
    input_file_name=sys.argv[1]
else:
    # input_file_name = './test_input.txt'
    input_file_name = './test_input_all_ordered.txt'


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
                    a = l[1].strip().split(' | ')
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
                    if (w_type == 'text_weight'):
                        if q_word[0].has_vector and s_word[0].has_vector:
                            if q_word[0].similarity(s_word[0]) > weight_dict['threshold']:
                                if q_word[0].text == bump_word and q_word[0].similarity(s_word[0]) > .75:
                                    b_weight = weight_dict["bump_weight"]
                                else:
                                    b_weight = 1

                                # # This is here just to gain some insight into word pairs
                                # # and their similarity scores
                                # # Perhaps if words' similarity scores fall below a certain
                                # # threshold then we don't allow it to contribute to the
                                # # overall context weight?  Also, ignore negative sim scores?
                                # sim = q_word[0].similarity(s_word[0])
                                # w_pair = q_word[0].text + ' ' + s_word[0].text
                                # if w_pair not in sims:
                                #     sims.append(w_pair)
                                #     print(w_pair, sim)

                                curr_context_weight += q_word[0].similarity(s_word[0]) * weight_dict[w_type] * b_weight
                    elif (w_type == 'pos_weight'):
                        curr_attr = attribute_dict[q_type]['POS']
                        if(s_word[0].pos_ in curr_attr):
                            curr_context_weight += curr_attr[s_word[0].pos_] * weight_dict[w_type]

        #context level comparisons
        if('ent_weight' in weight_dict):
            entities = []
            # Check if current context contains any entities
            context_start_idx = context_words[0][0].idx
            context_end_idx = context_words[-1][0].idx + len(context_words[-1][0].text) - 1
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
                    curr_context_weight += curr_attr[ent.label_] * weight_dict['ent_weight']
        if curr_context_weight > best_context_weight:
            best_context_weight = curr_context_weight
            best_context = context_words
    return best_context, best_context_weight

def get_sentence_weight(sentence, question_data, orig_story, attribute_dict, q_type, weight_dict, bump_word, q_words):
    sentence_weight = 0
    for q_word in question_data:
        if q_word[0].text.lower() in q_words:
            continue
        for s_word in sentence:
            if s_word.pos_ == 'PUNCT'or s_word.pos_ == "SPACE":
                continue
            for w_type in weight_dict:
                if(w_type == 'text_weight'):
                    if q_word[0].has_vector and s_word.has_vector:
                        if q_word[0].similarity(s_word) > 0:
                            if q_word[0].text == bump_word and q_word[0].similarity(s_word) > .75:
                                b_weight = weight_dict["bump_weight"]
                            else:
                                b_weight = 1
                            sentence_weight += q_word[0].similarity(s_word) * weight_dict[w_type] * b_weight
                elif(w_type == 'pos_weight'):
                    curr_attr = attribute_dict[q_type]['POS']
                    if(s_word.pos_ in curr_attr):
                        sentence_weight += curr_attr[s_word.pos_] * weight_dict[w_type]

    #context level comparisons
    if('ent_weight' in weight_dict):
        entities = []
        # Check if current sentence contains any entities
        context_start_idx = sentence[0].idx
        context_end_idx = sentence[-1].idx + len(sentence[-1].text) - 1
        for ent in orig_story.ents:
            start_idx = ent.start_char
            end_idx = ent.end_char
            if start_idx >= context_start_idx and end_idx <= context_end_idx:
                entities.append(ent)
            elif start_idx > context_end_idx:
                break
        curr_attr = attribute_dict[q_type]['ENT']
        for ent in entities:
            if ent.label_ in curr_attr:
                sentence_weight += curr_attr[ent.label_] * weight_dict['ent_weight']

    sentence_weight = sentence_weight / len(sentence)
    return sentence_weight

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

            else:
                q2 = w.lower() + ' ' + nlp_q[i + 1].text

            # Further split our biggest category
            # ROOT 'do' implies answer is VERB-based
            # Otherwise, more often than not, answer is NOUN-based
            if q2 == 'what did':
                for t in nlp_q:
                    if t.dep_ == 'ROOT':
                        rw = t.text
                        if rw == 'do':
                            q2 += ' ' + rw
                        elif rw == 'eat':
                            q2 += ' ' + rw
                        else:
                            increase_sim_weight = True
                        break
            
            if q2=="who is":
                if "PERSON" in [e.label_ for e in nlp_q.ents]:
                    q2+= ' ' + 'identity'
                else:
                    q2+=' '+'individual'

            if q2 not in list(q_2word_counts.keys()):
                # q_2word_counts[q2] = 1
                q_2word_counts[q2] = {}
                q_2word_counts[q2]['Count'] = 1
                q_2word_counts[q2]['ENT'] = {}
                # q_2word_counts[q2]['NP']= []
                q_2word_counts[q2]['POS'] = {}
                q_2word_counts[q2]['Avg Ans Len'] = a_lens  # Count lengths for now, take average at the end
                if increase_sim_weight:
                    q_2word_counts[q2]['Inc Sim Weight'] = True  # increase the similarity weight of the bump word
                else:
                    q_2word_counts[q2]['Inc Sim Weight'] = False
            else:
                # q_2word_counts[q2] += 1
                q_2word_counts[q2]['Count'] += 1
                q_2word_counts[q2]['Avg Ans Len'].extend(a_lens)


            for a in nlp_a:

                # THIS CAN BE USEFUL FOR DEBUGGING
                # if q2 == 'when did':
                #     for t in nlp_q:
                #         if t.dep_ == 'ROOT':
                #             rw = t.text
                #             break
                #     print(nlp_q.text)
                #     print(a.text)
                #     print('')

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

            # SPECIAL CASES
            if q_type == 'what did':
                for t in question:
                    if t.dep_ == 'ROOT':
                        rw = t.text
                        if rw == 'do':
                            q_type += ' ' + rw
                        else:  # not every category needs a bump word, but this is where they get assigned
                            bump_word = rw
                        break
            if q_type=="who is":
                if "PERSON" in [e.label_ for e in question.ents]:
                    q_type+= ' ' + 'identity'
                else:
                    q_type+=' '+'individual'                
            if q_type not in list(q_2word_counts.keys()):
                q_type = token.lower() + ' ' + question[i + 1].pos_
                if q_type not in list(q_2word_counts.keys()):
                    q_type = 'Generic'
            if q_2word_counts[q_type]['Inc Sim Weight']:
                if q_type != 'what did':
                    bump_word = question[i + 1].text
            return q_type, bump_word
    return "Generic", bump_word

leading_punct=',:;.!?\'\"({)}'
trailing_punct=',:;.!?\'\"({)}$'
def get_fscore(response, key):
    r=response.split()
    response=[]
    for y in r:
        adj=y.lower().lstrip(leading_punct).rstrip(trailing_punct)
        if adj != '':
            response.append(adj)
    best_fm=0
    b_recall=0
    b_prec=0
    for i,a in enumerate(key):
        asdf=a.split()
        ans=[]
        for x in asdf:
            adjx=x.lower().lstrip(leading_punct).rstrip(trailing_punct)
            if adjx != '':
                ans.append(adjx)
        correct = 0
        cwords = set()
        for wrd in response:
            w = wrd.lower().strip()
            if w in ans:
                if w not in cwords:
                    correct += 1
                    cwords.add(w)
        recall = correct/len(ans)
        prec = correct/len(response)
        fm = 0
        if recall+prec == 0:
            fm = 0
        else:
            fm =(2*recall*prec)/(recall+prec)
        if(fm > best_fm):
            best_fm = fm
            best_fm_ind = i
            b_recall = recall
            b_prec = prec
    return best_fm,b_prec, b_recall


def ent_trim(q_type, sentences, ent_dict):
    sorted_ent = [k for k, v in sorted(ent_dict[q_type]['ENT'].items(), key=lambda item: item[1], reverse=True)]
    ents_ans = []
    for i in range(len(sorted_ent)):
        k = sorted_ent[i]
        for score in sentences:
            for ent in sentences[score].ents:
                if ent.label_ == k:
                    ents_ans.append(ent)
            if len(ents_ans) > 0:
                s = ''
                for e in ents_ans:
                    s += e.text + ' '
                s = s.rstrip()
                return nlp(s)
                # return s
    return sorted_sents[list(sorted_sents.keys())[0]]


def why_bec_trim(sentence):
    return nlp('because' + sentence.text.split('because', 1)[1])


def get_biggest_pps(sentence):
    # Find prepositional phrases
    pps = []
    pps_text = []
    for token in best_sentence:
        if token.pos_ == 'ADP':
            pp = [tok for tok in token.subtree]
            pps.append(pp)
            pp_text = ' '.join([tok.orth_ for tok in token.subtree])
            pps_text.append(pp_text)
    # Check if pp is a subset of a larger pp
    i_del = []
    for i in range(len(pps_text)):
        for j in range(len(pps_text)):
            if i == j:
                continue
            else:
                if pps_text[i] in pps_text[j]:  # this pp in part of a larger one
                    i_del.append(i)
    # filter pps to only contain parent pps
    filtered_pps = []
    for i in range(len(pps)):
        if i not in i_del:
            filtered_pps.append(pps[i])
    return filtered_pps


def when_did_trim(sentence, pps):
    # First, try to return pps with DATEs in them
    best_pps = []
    for pp in pps:
        for token in pp:
            if token.ent_type_ == 'DATE':
                pp = ' '.join([t.text for t in pp])
                best_pps.append(pp)
                break
    if len(best_pps) > 0:
        return ' '.join(best_pps)

    # Next, try to return DATEs
    best_ents = []
    for ent in sentence.ents:
        if ent.label_ == 'DATE':
            best_ents.append(ent.text)
    if len(best_ents) > 0:
        return ' '.join(best_ents)

    return sentence.text

# ===========================
# ===========================

#######Load Data####### these are test sets
stories = {}
questions = {}
extra_questions={}
extra_stories={}
# Original data
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

# # Extra data
for fname in os.listdir(os.getcwd() + '/extra-data'):
    if '.answers' in fname:
        id = fname.split('.answers')[0]
        question_data, _ = load_QA('extra-data/' + id + '.answers')
        extra_questions[id] = question_data
    else:
        id = fname.split('.story')[0]
        story_data = load_story('extra-data/' + id + '.story')
        extra_stories[id] = story_data


#######yper parameters#######
# k = 5
weights = {"TEXT": .1, "POS": .5, "ENT": 1}
bump_weight = 2  # == 1 does nothing, should be greater than 1
# q_words = ['who', 'what', 'when', 'where', 'why', 'how', 'whose', 'which', 'did', 'are']  # couple weird ones here
####################################
filter_pos_tags = ['PUNCT', 'DET', 'SPACE']
stop_words = nlp.Defaults.stop_words
q_words = ['who', 'what', 'when', 'where', 'why', 'how', 'whose', 'which']
####################################


#######Build Dictionary for Question Types#######
q_2word_counts = np.load('./attribute_dictionary_testing', allow_pickle=True)

id_to_type = {}  # link q to type
q_type_set = set()
qtype_to_id = {}
qid_to_sid = {}
qid_to_vecq = {}
qid_to_answ = {}
qid_to_bump = {}
qid_to_orig_s = {}
qtype_counts={}
for story_id in list(questions.keys()):
    story_qa = questions[story_id]
    # story = stories[story_id]['TEXT']
    # vectorized_s = vectorize_list(filtered_s_text)
    for question_id in list(story_qa.keys()):
        question = story_qa[question_id]['Question']
        answers = story_qa[question_id]['Answer']
        nlp_a = [nlp(a) for a in answers]

        q_type, bw = get_q_type(nlp(question), q_words)
        if q_type not in qtype_counts:
            qtype_counts[q_type]=0
        qtype_counts[q_type]+=1
        #q_type = get_q_words_count(nlp(question), nlp_a)

        # if q_type not in q_2word_counts:
        #     tmp = nlp(q_type)
        #     q_type = tmp[0].text + ' ' + tmp[1].pos_
        #     if q_type not in q_2word_counts:
        #         q_type = 'Generic'

        id_to_type[question_id] = q_type
        q_type_set.add(q_type)

extra_qs_types={}
extra_st_types={}
extra_qids={}
for story_id in list(extra_questions.keys()):
    story_qa = extra_questions[story_id]
    # story = stories[story_id]['TEXT']
    # vectorized_s = vectorize_list(filtered_s_text)
    for question_id in list(story_qa.keys()):
        question = story_qa[question_id]['Question']
        answers = story_qa[question_id]['Answer']
        nlp_a = [nlp(a) for a in answers]

        q_type, bw = get_q_type(nlp(question), q_words)
        if q_type not in extra_qs_types:
            extra_qs_types[q_type]=[]
            extra_st_types[q_type]=[]
        extra_qs_types[q_type].append(story_id)
        extra_st_types[q_type].append(story_id)
        #q_type = get_q_words_count(nlp(question), nlp_a)

        # if q_type not in q_2word_counts:
        #     tmp = nlp(q_type)
        #     q_type = tmp[0].text + ' ' + tmp[1].pos_
        #     if q_type not in q_2word_counts:
        #         q_type = 'Generic'

        id_to_type[question_id] = q_type
        q_type_set.add(q_type)


for typ in qtype_counts:
    if qtype_counts[q_type]<53:
        if typ in extra_qs_types:
            for qid in extra_qs_types[typ]:
                if(qtype_counts[typ]==53):
                    break
                questions[qid]=extra_questions[qid]
                stories[qid]=extra_stories[qid]
                qtype_counts[typ]+=1

# print('[')
# for typ in qtype_counts:
#     print('\''+typ+'\''+',')

# print(']')

# new_q2 = {}
# for k1 in q_2word_counts.keys():
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
#
# # Add a 'Generic' feature to our q_2word_counts, a weighted avg of all other features
# # This is in case we come across a question we've never seen
# for k in q_2word_counts.keys():
#     count += q_2word_counts[k]['Count']
# nkeys = len(list(q_2word_counts.keys()))
# gen_keys = ['ENT', 'POS']
# generic_count = {'ENT': {}, 'POS': {}, 'Avg Ans Len': 5, 'Inc Sim Weight': False}
# for k1 in q_2word_counts.keys():
#     cw = q_2word_counts[k1]['Count'] / count
#     for k2 in gen_keys:
#         for k3 in q_2word_counts[k1][k2].keys():
#             if k3 not in generic_count[k2]:
#                 generic_count[k2][k3] = q_2word_counts[k1][k2][k3] * cw
#             else:
#                 generic_count[k2][k3] += q_2word_counts[k1][k2][k3] * cw
# q_2word_counts['Generic'] = generic_count



for story_id in list(questions.keys()):
    story_qa = questions[story_id]
    story = stories[story_id]['TEXT']

    # story_sents = list(nlp2(story).sents)
    # test1 = [token for token in nlp(story_sents[0].text)]
    # test2 = [token for token in nlp(story)]
    tagged_text = [[token, i] for i, token in enumerate(nlp(story))]
    filtered_s = filter_by_POS(tagged_text, filter_pos_tags)
    filtered_s = filter_by_stopwords(filtered_s, stop_words)

    for question_id in list(story_qa.keys()):

        question = story_qa[question_id]['Question']
        answer = story_qa[question_id]['Answer']
        tagged_q = [[token, i] for i, token in enumerate(nlp(question))]
        filtered_q = filter_by_POS(tagged_q, filter_pos_tags)
        filtered_q = filter_by_stopwords(filtered_q, stop_words)
        q_type, bump = get_q_type(nlp(question), q_words)  # q_type, bump_word = get_q_type(nlp(question), q_words)
        qid_to_bump[question_id] = bump

        # vectorized_q = vectorize_list(filtered_q)
        # vectorized_s = vectorize_list(filtered_s_text)

        if q_type not in qtype_to_id:
            qtype_to_id[q_type] = []
        qtype_to_id[q_type].append(question_id)
        qid_to_orig_s[question_id] = nlp(story)
        qid_to_sid[question_id] = filtered_s
        qid_to_vecq[question_id] = filtered_q
        correct_ans = story_qa[question_id]['Answer']
        # correct_filt = []
        # for resp in correct_ans:
        #     f_response = [token.text for token in nlp(resp) if token.pos_ != 'PUNCT' and token.pos_ != 'SPACE']
        #     correct_filt.append(f_response)
        #     # correct_filt.append(filter_by_POS(nlp(resp), ['PUNCT', 'SPACE'])[0])
        # to_check = []
        # for i,resp in enumerate(correct_filt):
        #     to_check.append([])
        #     for w in (resp):
        #         to_check[i].append(w.lower().strip())
        qid_to_answ[question_id] = correct_ans



######tune#######
best_params={} #per q id type
for typ in q_type_set:
    if typ not in best_params:
        best_params[typ]={"text_weight":0,"pos_weight":0, "ent_weight":0,'k':0,'bump_weight':0,'threshold':0}

# for typ in best_params:
#     for wt in best_params[typ]:
#         for w in range(0,2,.1):
#             if(wt != "k"):
#                 best_params[typ][wt][w]=[]
#         for w in range(0,11,1):
#             if(wt=="k"):
#                 best_params[typ][wt][w]=[]
best_params_per_story={}

#######Load Data####### these are test sets
# stories = {}
# questions = {}
# for fname in os.listdir(os.getcwd() + '/data'):
#     id = fname.split('.')[0]
#     story_data = load_story('data/' + id + '.story')
#     question_data, _ = load_QA('data/' + id + '.answers')
#     stories[id] = story_data
#     questions[id] = question_data
# which PROPN
# who did
# when PRON
# what were
# why is
# whose NOUN
# where does
# what do
# what does
# which NOUN
# where VERB
# what was
# what kind
# what ADJ
# where ADP
# how ADV
# which NUM
# how does
# what did
# why VERB
# why AUX
# how AUX
# what are
# how did
# where was
# where AUX
# why does
# why was
# when was
# who ADV
# what PROPN
# what ADP
# what ADV
# what time
# 'where is',
# 'who is',
# 'what AUX',
# 'who VERB',
# 'how ADJ',
# 'how much',
# 'what NOUN',
# 'how many',
# 'when did',
# 'why did',
# 'who was',
# 'where did',
# 'who AUX',
# 'how VERB',
# 'what happened',
# 'how old',
# 'when VERB',
# 'what VERB',
# 'when AUX',
# 'how long',
# 'what NUM',
# 'what is',
# 'what type',
# 'who did',
# 'which PROPN',
# to_use_qtype=[
# 'why PROPN',
# 'what day',
# 'what color',
# 'which of',
# 'when PROPN',
# 'what animal',
# 'who DET',
# 'who CCONJ',
# 'whose PROPN',
# 'which VERB',
# 'which AUX',
# ]
# to_use_qtype=[
#     'who is identity',
#     'who is individual'
# ]

to_use_qtype_alec=[
'where is',
'who is individual',
'what AUX',
'who VERB',
'how ADJ',
'how much',
'what NOUN',
'how many',
'when did',
'why did',
'who was',
'where did',
'who AUX',
'why do',
'how VERB',
'who is identity',
'what happened',
'how old',
'when VERB',
'what VERB',
'when AUX']
to_use_qtype_torin=[
'how long',
'what NUM',
'what is',
'what type',
'where VERB',
'what does',
'whose NOUN',
'how ADV',
'when was',
'what do',
'how did',
'what did',
'why AUX',
'what ADJ',
'which NOUN',
'why VERB',
'where does',
'why was',
'where AUX',
'how AUX',
'where ADP',
'why does',
'why is',
'how does',
'what are',
'what was',
'what kind',
'where was',
'what were',
'what did do',
'who did',
'which PROPN',
'which NUM',
'when PRON',
'who ADV',
'Generic',
'what ADV',
'what ADP',
'what time'
]
for qtype_i in to_use_qtype_torin:
    print("length of set is ", len(qtype_to_id[qtype_i]))
    best_w_t=0
    best_w_p=0
    best_w_e=0
    best_w_k=0
    # best_ofm=0
    best_ofr = 0
    best_w_b=0 
    # best_fm_sum=0
    best_recall_sum = 0
    best_thresh=0
    j=0
    print("for qtype", qtype_i)

    for curr_weight_t in [1,2,4,8]:
        for curr_weight_p in [.5, 1, 2]:
            for curr_weight_e in [1,2,4,8]:
                curr_k = math.ceil(q_2word_counts[qtype_i]['Avg Ans Len'] / 2)
                for t in [0,.25,.5]:
                    for curr_b in [2,4]:
                        # curr_fm_sum = 0
                        curr_recall_sum = 0
                        print('percent done per qtype: ', (j/288.0)*100)
                        j+=1
                        for question_i in qtype_to_id[qtype_i]:
                            filtered_s = qid_to_sid[question_i]
                            filtered_q = qid_to_vecq[question_i]
                            orig_story = qid_to_orig_s[question_i]
                            answer=qid_to_answ[question_i]
                            # best_context = get_best_context_w_weight(vectorized_s, vectorized_q, q_2word_counts, curr_k, qtype_i, {"TEXT": curr_weight_t, "POS": curr_weight_p, "ENT": curr_weight_e, "BUMP":curr_b},qid_to_bump[question_id])
                            best_context, ents = get_best_context_w_weight(filtered_s, filtered_q, orig_story, q_2word_counts, curr_k, qtype_i,{"text_weight": curr_weight_t, "pos_weight": curr_weight_p, "ent_weight": curr_weight_e, "bump_weight":curr_b, 'threshold':t}, qid_to_bump[question_id], q_words)
                            # to_check = qid_to_answ[question_i]
                            sents_text = []
                            sents = []
                            for w in best_context:
                                if w[0].sent.text not in sents_text:
                                    sents_text.append(w[0].sent.text)
                                    sents.append(w[0].sent)
                            # Find the sentence with the highest per-word weight
                            best_weight = 0
                            sorted_sents = {}  # not sorted until they get passed to ent_sent_trim
                            for i,s in enumerate(sents):
                                st = [token for token in s]
                                sentence_weight = get_sentence_weight(st, filtered_q, nlp(story), q_2word_counts, q_type, {"TEXT": curr_weight_t, "POS": curr_weight_p, "ENT": curr_weight_e, "BUMP":curr_b}, qid_to_bump[question_id], q_words)
                                sorted_sents[str(sentence_weight)] = s
                                if i == 0:
                                    best_sentence = s
                                    best_weight = sentence_weight
                                elif sentence_weight > best_weight:
                                    best_weight = sentence_weight
                                    best_sentence = s
                            sorted_sents = dict(sorted(sorted_sents.items(), reverse=True))

                            best_context_text = ''
                            for t in best_context:
                                best_context_text += t[0].text + ' '
                            try:
                                fscore, prec, best_recall=get_fscore(best_sentence.text, answer)
                            except TypeError:
                                fscore, prec, best_recall=get_fscore(best_context, answer)
                            # best_fm_ind = 0
                            # best_fm = 0
                            # # best_recall_ind = 0
                            # # best_recall = 0
                            # # response = word_tokenize(best_context.text)
                            # # # response = [token for token in best_context]  TODO: remove nltk dep
                            # # # b_recall = 0
                            # # # b_prec = 0
                            # # for i,resp in enumerate(to_check):
                            # #     correct = 0
                            # #     cwords = set()
                            # #     for wrd in response:
                            # #         w = wrd.lower().strip()
                            # #         if w in to_check[i]:
                            # #             if w not in cwords:
                            # #                 correct += 1
                            # #                 cwords.add(w)
                            # #     recall = correct/len(to_check[i])
                            # #     # prec = correct/len(response)
                            # #     # fm = 0
                            # #     # if recall+prec == 0:
                            # #     #     fm = 0
                            # #     # else:
                            # #     #     fm =(2*recall*prec)/(recall+prec)
                            # #     # if(fm > best_fm):
                            # #     #     best_fm = fm
                            # #     #     best_fm_ind = i
                            # #     #     b_recall = recall
                            # #     #     b_prec = prec
                                # # if (recall > best_recall):
                                # #     best_recall = recall
                                # #     best_recall_ind = i

                            # curr_fm_sum += best_fm
                            curr_recall_sum += best_recall
                        ############done wit q loop###############
                        # if(curr_fm_sum >= best_fm_sum):
                        #     best_w_t = curr_weight_t
                        #     best_w_p = curr_weight_p
                        #     best_w_e = curr_weight_e
                        #     best_w_k = curr_k
                        #     best_ofm = best_fm
                        #     best_w_b = curr_b
                        #     best_fm_sum = curr_fm_sum
                        if (curr_recall_sum > best_recall_sum):
                            best_w_t = curr_weight_t
                            best_w_p = curr_weight_p
                            best_w_e = curr_weight_e
                            best_w_k = curr_k
                            best_ofr = best_recall
                            best_w_b = curr_b
                            best_recall_sum = curr_recall_sum
                            best_thresh=t
                            print('for weights', curr_weight_t, curr_weight_p, curr_weight_e, curr_k, curr_b,t)
                            # print(best_context, response)
                            print("recall= ", best_recall_sum)
                            print('\n')
                        
                    
                
            
     #####done_with_qtype_loop###########  
   
    best_params[qtype_i]["text_weight"]=(best_w_t)
    best_params[qtype_i]["pos_weight"]=(best_w_p)
    best_params[qtype_i]["ent_weight"]=(best_w_e)
    best_params[qtype_i]["k"]=(best_w_k)
    best_params[qtype_i]["bump_weight"]=(best_w_b)
    best_params[qtype_i]['threshold']=t
    try: 
        f = open('tuned_weights_TORIN', 'wb')
        pickle.dump(best_params, f) 
        f.close()
    except: 
        print("AAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHH Something went wrong")
    print("DONE WITH Q")
    # print('for type',qtype_i)
    # # print(best_context, response)
    # # print("overall best fmeasure= ",best_fm_sum, 'num of qs', len(qtype_to_id[qtype_i]))
    # print("overall best recall= ", best_recall_sum, 'num of qs', len(qtype_to_id[qtype_i]))
    # if len(qtype_to_id[qtype_i])>0:
    #     # print('ave', best_fm_sum/len(qtype_to_id[qtype_i]))
    #     print('ave', best_recall_sum / len(qtype_to_id[qtype_i]))

print('done')
    
# averages={}
# for typ in best_params:
#     if typ not in averages:
#         averages[typ]={}
#     for wtyp in best_params[typ]:
#         if(len(best_params[typ][wtyp])>0):
#             average=mean(best_params[typ][wtyp])
            
#             averages[typ][wtyp]=round(average, 3)
# print(averages)  
# try: 
#     f = open('tuned_weights_shortd', 'wb') 
#     pickle.dump(averages, f) 
#     f.close() 
  
# except: 
#     print("Something went wrong")
#TODO: write function for getting entire context not filtered
#function for output
#cleanup k issue
#check with scoring mechanism
#remove stop dep for nltk
#remove wordnet dep
#write install script
#write README
# exclude entity words that are in the question?