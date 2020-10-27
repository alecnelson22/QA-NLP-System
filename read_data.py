import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


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


# Returns list of tuples (word, POS) given a string of text
def get_POS_tags(text):
    tokenized = nltk.word_tokenize(text)
    return nltk.pos_tag(tokenized)


# Grabs words k to the left and k to the right of word at target index t_idx
def get_context_words(text, k, t_idx, split=True):
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
        if t_idx+1+k < len(text):
            end = t_idx+1+k
        else:
            end = len(text)
        r_words = text[t_idx+1:end]
        words = l_words + r_words
    return words


# Finds and returns the context with size k from a story text that
# contains the highest number of words from a question text
def get_best_context(story, question, k):
    best_context_matches = 0
    for t_idx in range(len(story)):
        context_words = get_context_words(story, k, t_idx)
        curr_context_matches = 0
        for q_word in question.split():
            if q_word in context_words:
                curr_context_matches += 1
        if curr_context_matches > best_context_matches:
            best_context_matches = curr_context_matches
            best_context = context_words
    return best_context


# ===========================
# s = load_story_sentences('data/1999-W02-5.story')

stories = {}
questions = {}

id = '1999-W02-5'
story_data = load_story('data/' + id + '.story')
question_data = load_QA('data/' + id + '.answers')

stories[id] = story_data
questions[id] = question_data

# Find the best context- section of story words with most overlap of questions words
k = 5
for story_id in list(questions.keys()):
    story_qa = questions[story_id]
    for question_id in list(story_qa.keys()):
        question = story_qa[question_id]['Question']
        story = stories[story_id]['TEXT']
        best_context = get_best_context(story, question, k)

# Get POS tags for stories
for k in list(s.keys()):
    text = s[k]['TEXT']
    pos = get_POS_tags(text)
