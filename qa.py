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
                            if q_word[0].similarity(s_word[0]) > 0:
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
                if q2 == 'where did':
                    for t in nlp_q:
                        if t.dep_ == 'ROOT':
                            rw = t.text
                            break
                    # print(nlp_q.text)
                    # print(a.text)
                    # print('')

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


# checks if there is a single noun chunk followed by a verb in a sentence, returns them
def get_ncv(sentence):
    chunks = [token for token in sentence.noun_chunks]
    ncv = []
    for c in chunks:
        f = sentence[-1]
        if c.end < sentence[-1].i:
            v = sentence[c.end]
            if sentence[c.end].pos_ == 'VERB':
                ncv.append([c, v])
    if len(ncv) > 0:
        return ncv
    return None


def ncv_story_search(story, ncv):
    sents = []
    for pair in ncv:
        nc = pair[0]
        v = pair[1]
        for chunk in story.noun_chunks:
            if nc.text == chunk.text:
                # check verb similarty
                w = story[chunk.end]
                sim = v.similarity(story[chunk.end])
                if sim > .9:
                    sents.append(w.sent)
    if len(sents) > 0:
        return sents
    return None

# ===========================
# ===========================

cats = {}

#######Load Data####### these are test sets
stories = {}
questions = {}
test_answer_key = {}
# ids=set()
for fname in os.listdir(os.getcwd() + '/data'):
    id = fname.split('.')[0]
    story_data = load_story('data/' + id + '.story')
    question_data, _ = load_QA('data/' + id + '.answers')
    stories[id] = story_data
    questions[id] = question_data
    test_answer_key[id] = question_data

#     ids.add(id)

# Test set 1
for fname in os.listdir(os.getcwd() + '/testset1'):
    id = fname.split('.')[0]
    story_data = load_story('testset1/' + id + '.story')
    question_data, _ = load_QA('testset1/' + id + '.answers')
    stories[id] = story_data
    questions[id] = question_data
    test_answer_key[id] = question_data
    # ids.add(id)
    # test_answer_key[id] = question_data

for fname in os.listdir(os.getcwd() + '/extra-data'):
    if '.answers' in fname:
        id = fname.split('.answers')[0]
        question_data, _ = load_QA('extra-data/' + id + '.answers')
        questions[id] = question_data
    else:
        id = fname.split('.story')[0]
        story_data = load_story('extra-data/' + id + '.story')
        stories[id] = story_data
        # ids.add(id)

# sids=sorted(ids)
# for id in ids:
#     print(id)

# asdf 

#######yper parameters#######
# k = 5
# default_weights = {"TEXT": 1, "POS": .25, "ENT": 1, "BUMP": 1, 'K': 5}
default_weights = {"text_weight": 1, "pos_weight": .25, "ent_weight": 1, "bump_weight": 1, 'k': 5}
# default_k=4
# bump_weight = 2  # == 1 does nothing, should be greater than 1
# q_words = ['who', 'what', 'when', 'where', 'why', 'how', 'whose', 'which', 'did', 'are']  # couple weird ones here
####################################
filter_pos_tags = ['PUNCT', 'DET', 'SPACE']
# filter_pos_tags = ['PUNCT', 'DET', 'SPACE', 'ADV', 'AUX', 'PRON', 'ADP']

stop_words = nlp.Defaults.stop_words
# q_words = ['who', 'what', 'when']
q_words = ['who', 'what', 'when', 'where', 'why', 'how', 'whose', 'which']

sims = []

####################################

######Build Dictionary for Question Types#######
q_2word_counts = {}  # attribute dictionary
id_to_type = {}  # link q to type
for story_id in list(questions.keys()):
    story_qa = questions[story_id]
    for question_id in list(story_qa.keys()):
        question = story_qa[question_id]['Question']
        answers = story_qa[question_id]['Answer']
        tokenized_q = [token.text for token in nlp(question)]
        nlp_a = [nlp(a) for a in answers]
        q_type = get_q_words_count(nlp(question), nlp_a)
        id_to_type[question_id] = q_type  # TODO: In theory, this is just a training step.  Thus, id_to_type needs to be removed, since it is referenced in ##run##
# q_2word_counts = {k: v for k, v in sorted(q_2word_counts.items(), key=lambda item: item[1], reverse=True)}

new_q2 = {}
for k1 in q_2word_counts.keys():
    # if q_2word_counts[k1] < 10:
    #     new_key = [token for token in nlp(k1)]
    #     new_key = new_key[0].text + ' ' + new_key[1].pos_
    #     if new_key not in new_q2:
    #         new_q2[new_key] = q_2word_counts[k1]
    #     else:
    #         new_q2[new_key] += q_2word_counts[k1]
    # else:
    #     new_q2[k1] = q_2word_counts[k1]
    if q_2word_counts[k1]['Count'] < 10:
        new_key = [token for token in nlp(k1)]
        new_key = new_key[0].text + ' ' + new_key[1].pos_
        if new_key not in new_q2:
            new_q2[new_key] = q_2word_counts[k1]
        else:
            for k2 in q_2word_counts[k1].keys():
                if k2 == 'POS' or k2 == 'ENT':
                    for k3 in q_2word_counts[k1][k2].keys():
                        if k3 not in new_q2[new_key][k2].keys():
                            new_q2[new_key][k2][k3] = q_2word_counts[k1][k2][k3]
                        else:
                            new_q2[new_key][k2][k3] += q_2word_counts[k1][k2][k3]
                elif k2 == "Inc Sim Weight":
                    new_q2[new_key][k2] = q_2word_counts[k1][k2]
                else:
                    new_q2[new_key][k2] += q_2word_counts[k1][k2]
    else:
        new_q2[k1] = q_2word_counts[k1]
q_2word_counts = new_q2
# q_2word_counts = {k: v for k, v in sorted(q_2word_counts.items(), key=lambda item: item[1], reverse=True)}
# np.save('sorted_qtypes', q_2word_counts)
get_avg_ans_len()

# Normalize q_2word_counts values
norm_keys = ['ENT', 'POS']  # values to normalize
for q2 in q_2word_counts.keys():
    for k in norm_keys:
        count = 0
        for item in q_2word_counts[q2][k].keys():
            count += q_2word_counts[q2][k][item]
        for item in q_2word_counts[q2][k].keys():
            q_2word_counts[q2][k][item] = q_2word_counts[q2][k][item] / count

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
f = open('attribute_dictionary_testing', 'wb')
pickle.dump(q_2word_counts, f)
f.close()
# asdf
# #######LOAD INPUT FOR TESTING #################
q_2word_counts=np.load('./attribute_dictionary_testing', allow_pickle=True)
loaded_weights=np.load('./tuned_weights_TEST_ALL', allow_pickle=True)
ent_dict=np.load('./ent_prob_dict', allow_pickle=True)
big_q_counts=np.load('./biggest_qtype_counts', allow_pickle=True)

count = 0

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
type_to_score={}
type_to_info={}
weights=[]
scores=[]
######run#######
for story_id in ordered_ids:
    story_qa = test_questions[story_id]
    story = test_stories[story_id]['TEXT']
    tagged_text = [[token, i] for i, token in enumerate(nlp(story))]
    filtered_s = filter_by_POS(tagged_text, filter_pos_tags)
    filtered_s = filter_by_stopwords(filtered_s, stop_words)
    # vectorized_s = vectorize_list(filtered_s_text)

    for question_id in ordered_qs[story_id]:
        question = story_qa[question_id]['Question']
        answer = test_answer_key[story_id][question_id]['Answer']
        # q_type = id_to_type[question_id]  # TO DO: This takes information from a pre-processing step, thus should be removed
        q_type, bump_word = get_q_type(nlp(question), q_words)
        tagged_q = [[token, i] for i, token in enumerate(nlp(question))]
        filtered_q = filter_by_POS(tagged_q, filter_pos_tags)
        filtered_q = filter_by_stopwords(filtered_q, stop_words)
        # vectorized_q = vectorize_list(filtered_q_text)

        # k = math.ceil(q_2word_counts[q_type]['Avg Ans Len'] / 2)

        # q_type2 = q_type.split()
        # if len(q_type2)>1:
        #     if q_type2[1].islower():
        #         tmp = [token for token in nlp(q_type)]
        #         q_type2 = tmp[0].text + ' ' + tmp[1].pos_
        #     else:
        #         q_type2 = q_type2[0] + ' ' + q_type2[1]
        # else:
        #     q_type2=q_type2[0]
        #
        # used_weights = default_weights
        # if q_type2 in loaded_weights.keys():
        #     # if loaded_weights[q_type2]["TEXT"] > 0:
        #     #     used_weights = loaded_weights[q_type2]
        #     used_weights = loaded_weights[q_type2]

        used_weights = default_weights
        tmp = [token for token in nlp(q_type)]
        used_type=q_type
        if q_type != 'Generic':
            q_type2 = tmp[0].text + ' ' + tmp[1].pos_
            if q_type in loaded_weights.keys():
                used_weights = loaded_weights[q_type]
            elif q_type2 in loaded_weights.keys():
                used_weights = loaded_weights[q_type2]

        # Try finding signature (1 noun chunk followed by 1 verb) in question)
        # If it exists, try to find a highly similar match in story, before calling gbc

        # if q_type == 'where did':

        ncv = get_ncv(nlp(question.strip()))
        if ncv is not None:
            sents = ncv_story_search(nlp(story), ncv)
            if sents is None:
                # k = used_weights['k']
                k = 4
                best_context, weight = get_best_context_w_weight(filtered_s, filtered_q, nlp(story), q_2word_counts, k, q_type, used_weights, bump_word, q_words)
                # Find all sentences that are a part of the best context
                sents_text = []
                sents = []
                for w in best_context:
                    if w[0].sent.text not in sents_text:
                        sents_text.append(w[0].sent.text)
                        sents.append(w[0].sent)
        else:
            k = 4
            best_context, weight = get_best_context_w_weight(filtered_s, filtered_q, nlp(story), q_2word_counts, k,
                                                             q_type, used_weights, bump_word, q_words)
            # Find all sentences that are a part of the best context
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
            sentence_weight = get_sentence_weight(st, filtered_q, nlp(story), q_2word_counts, q_type, used_weights, bump_word, q_words)
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

        orig_sentence=best_sentence
        # print('Question: ', question,file=sys.stderr)
        # print('Best context: ', best_context_text,file=sys.stderr)
        # print('Best sentence: ', best_sentence,file=sys.stderr)
        # print('Actual: ', answer,file=sys.stderr )
        
        # k = math.ceil(q_2word_counts[used_type]['Avg Ans Len'] / 2)
        # best_sentence_tokens=[[token, i] for i, token in enumerate(best_sentence)]

        # print(filtered_s,file=sys.stderr)
        # print( best_sentence_tokens,file=sys.stderr)
        # best_response_res,_= get_best_context_w_weight(best_sentence_tokens, filtered_q, best_sentence, q_2word_counts, k, q_type, used_weights, bump_word, q_words)
        # best_response=''
        # for t in best_response_res:
        #     best_response += t[0].text + ' '
        # print('best response', best_response,file=sys.stderr)

        # CASE-BY-CASE SENTENCE TRIMMING
        # Entity-based sentence trim
        if q_type in ent_dict:
            best_sentence = ent_trim(q_type, sorted_sents, ent_dict)

        # why/because-based sentence trim
        elif q_type.split()[0] == 'why' and 'because' in best_sentence.text:
            best_sentence = why_bec_trim(best_sentence)

        #prep phrase-based trim
        elif q_type == 'when did':
            pps = get_biggest_pps(best_sentence)
            best_sentence = when_did_trim(best_sentence, pps)
        # elif q_type == 'who is':
        #     pps = get_biggest_pps(best_sentence)
            # best_sentence = when_did_trim(best_sentence, pps)
        print('Question: ', question,file=sys.stderr)
        print('Best context: ', best_context_text,file=sys.stderr)
        print('Best sentence: ', best_sentence,file=sys.stderr)
        print('Best original sentence: ', orig_sentence,file=sys.stderr)

        print('Actual: ', answer, '\n',file=sys.stderr)
        # fscore, prec, recall= get_fscore(best_sentence.text,answer )
        # print('fscore: ', fscore, file=sys.stderr)
        # print('prec/recal: ', prec,recall,'\n', file=sys.stderr)

#         # print(question,file=sys.stderr)
#         # print(story_qa[question_id]['Answer'],file=sys.stderr)
#         # print(best_context,file=sys.stderr)
#         # print('\n',file=sys.stderr)
#
        try:
            print(best_sentence +"\n", file=sys.stderr)
            print('QuestionID: '+question_id)
            print('Answer: ' + best_sentence + "\n")
        except TypeError:
            print('QuestionID: '+question_id)
            print('Answer: ' + best_sentence.text + "\n")
#         print('QuestionID: ' ,question, file=sys.stderr)
#         print('Answer: ', answer, file=sys.stderr)
        
        # weights.append(weight)
        # scores.append(recall)
        # if(q_type not in type_to_score):
        #     type_to_score[q_type]=[]
        # type_to_score[q_type].append(recall)
        # if q_type not in type_to_info:
        #     type_to_info[q_type]=[]
        # # sent_doc=best_sentence.as_doc()
        # print('ents',[e.label_ for e in best_sentence.ents], file=sys.stderr)
        # type_to_info[q_type].append({"question":question,"answer-key":answer, "best_sentence":best_sentence.text, "fscore":fscore,"precision":prec,'recall':recall, "best_sent_ents": [e.label_ for e in best_sentence.ents]})
        # break
    # break
#         # ans = ''
#         # for w in best_context:
#         #     ans += w[0].text + ' '
#         print('Answer: ' + best_context.text, file=sys.stderr)
#         # print('Entities in answer: ', [e.label_ for e in ents], '\n', file=sys.stderr)
# #         outputs.append([question_id, best_context])
# : 0.35000777000777, 'what VERB': 0.3704695767195767, 'who VERB': 0.394140989729225, 'who AUX': 0.398046398046398, 'what NOUN': 0.40522875816993464, 'what ADJ': 0.42045454545454547, 'when AUX': 0.45061949544708163, 'why VERB': 0.4865800865802163742690059}
# type_to_score_ave={}
# for typ in type_to_score:
#     mean=statistics.mean(type_to_score[typ])
#     print('for qtype', typ, "ave score is ",mean, file=sys.stderr)
#     type_to_score_ave[typ]=mean

# ordered={k: v for k, v in sorted(type_to_score_ave.items(), key=lambda item: item[1])}
# print(ordered, file=sys.stderr)

# f = open('summary_noextra_new', 'wb')
# pickle.dump(type_to_info, f) 
# f.close()

# f = open('qtype_recall_noextra_new', 'wb')
# pickle.dump(type_to_score, f) 
# f.close()
# for output in outputs:
#     print('QuestionID: '+output[0])
#     print('Answer: ' + output[1] + "\n")

    # plt.scatter(weights, scores)
    # plt.show()
#TODO: write function for getting entire context not filtered
#function for output
#cleanup k issue
#check with scoring mechanism
#remove stop dep for nltk
#remove wordnet dep
#write install script
#write README