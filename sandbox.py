import numpy as np 
import pickle
import statistics 
import os
import spacy
# from qa import *


# for fname in os.listdir(os.getcwd() + '/all_data'):
#     if '.answers' in fname:
#         id = fname.split('.answers')[0]
#     # story_data = load_story('all_data/' + id + '.story')
#     # question_data, _ = load_QA('all_data/' + id + '.answers')
#     # stories[id] = story_data
#     # questions[id] = question_data
#     # test_answer_key[id] = question_data
#         print(id)
# sasdf
print('loading data...')
# recalls=np.load('./qtype_recall', allow_pickle=True)
# summary=np.load('./summary', allow_pickle=True)
recalls=np.load('./qtype_recall_noextra', allow_pickle=True)
summary=np.load('./summary_noextra', allow_pickle=True)
# f = open('./summary')
# summary=pickle.load(f)
nlp = spacy.load("en_core_web_lg")  # make sure to use larger model!
print('finished')

# print(summary,recalls)
type_to_score_ave={}

for typ in recalls:
    mean=statistics.mean(recalls[typ])
    # print('for qtype', typ, "ave score is ",mean)
    type_to_score_ave[typ]=mean

ordered={k: v for k, v in sorted(type_to_score_ave.items(), key=lambda item: item[1])}
# print(ordered)

# {"question":question,"answer-key":answer, "best_sentence":best_sentence, "fscore":fscore,"precision":prec,'recall':recall}
bad_questions={}
count=0
count_bad=0
good_counts={}
bad_counts={}

for typ in ordered:
    print('for qtype', typ, 'ave recall is ', ordered[typ])
    bad_counts[typ]=0
    good_counts[typ]=0
    for q,quest in enumerate(summary[typ]):
        if summary[typ][q]['recall']<.8:
            if typ not in bad_questions:
                bad_questions[typ]=[]
            bad_questions[typ].append(summary[typ])
            count_bad+=1
            bad_counts[typ]+=1
        else:
            good_counts[typ]+=1
        count+=1
print('-------------')
for typ in ordered:
    if bad_counts[typ]==0:
        print('ratio of success/failure for qtype', typ,'is all above .8','len of set is ',len(summary[typ]))
    else:
        print('ratio of success/failure for qtype', typ,'is ',good_counts[typ]/bad_counts[typ],'len of set is ',len(summary[typ]))
print('length of all questions is ',count, 'below .8 recall len',count_bad )
print('----------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------')

# asdf
for typ in bad_questions:
    if(good_counts[typ]/bad_counts[typ]>=1):
        continue
    
    print('-------------------FOR TYPE ',typ,'ratio',good_counts[typ]/bad_counts[typ],'bad questions are -----------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')
    
    for q,quest in enumerate(bad_questions[typ]):
        ents=summary[typ][q]['best_sent_ents']
        print('---------------------------------')
        print('question',summary[typ][q]['question'])
        print('key ', summary[typ][q]['answer-key'])
        print('best_sentence', summary[typ][q]['best_sentence'],'\n')
        print('Entities in sentence: ', ents)
        print('--answer characteristics--')
        for ans in summary[typ][q]['answer-key']:
            nlp_key=nlp(ans)
            print('Entities in key: ', [e.label_ for e in nlp_key.ents])
        # print('')
        for ans in summary[typ][q]['answer-key']:
            nlp_key=nlp(ans)
            print('word/POS/dep in key are ', [(token.text, token.pos_,token.dep_) for token in nlp_key])
        print('') 
        nlp_q=nlp(summary[typ][q]['question'])   
        print('---question dependency info------ ')
        print('word/POS/dep in question are ', [(token.text,token.pos_,token.dep_)  for token in nlp_q])
        print('dep chunks text/root_dep in question are ', [(chunk.text, chunk.root.dep_) for chunk in nlp_q.noun_chunks])
        print('critical word/POS/dep in question are ', [(token.text,token.pos_,token.dep_) for token in nlp_q if(token.dep_=="ROOT" or token.dep_=="dobj" or token.dep_=="nsubj")])

        print('---------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

