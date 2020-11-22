import numpy as np 
import pickle

# my_dictl=np.load('./tuned_weights_long', allow_pickle=True)
# my_dict=np.load('./tuned_weights_short', allow_pickle=True)
# my_dicta=np.load('./tuned_weights_shorta', allow_pickle=True)
# my_dictb=np.load('./tuned_weights_shortf_B', allow_pickle=True)
# my_dictc=np.load('./tuned_weights_shortf_remaining', allow_pickle=True)
# my_dictwa=np.load('./tuned_weights_shortf_whenaux', allow_pickle=True)
# my_dictr2=np.load('./tuned_weights_shortf_whenaux', allow_pickle=True)
# my_dictd=np.load('./tuned_weights_shortd', allow_pickle=True)
# my_dictheyyy=np.load('./tuned_weights_how_ADJ', allow_pickle=True)
# who_adv=np.load('./tuned_weights_torin_whoadv', allow_pickle=True)
# whereverb=np.load('./tuned_weights_torin_whereverb', allow_pickle=True)
# whatnoun=np.load('./tuned_weights_alec_whatNOUN', allow_pickle=True)
# the_rest=np.load('./tuned_weights_the_rest', allow_pickle=True)
who_is=np.load('./tuned_weights_TORIN', allow_pickle=True)




added=[]

# my_final_dict=np.load('./tuned_weights_TEST_ALL', allow_pickle=True)
my_final_dict={}
for typ in who_is:
    # if who_is[typ]["text_weight"]!=0:
        if typ not in my_final_dict:
            my_final_dict[typ]={}

            print('for type ', typ)
            print('text_weight', who_is[typ]["text_weight"])
            print('pos_weight', who_is[typ]["pos_weight"])
            print('ent_weight', who_is[typ]["ent_weight"])
            print('k', who_is[typ]["k"])
            print('bump_weight', who_is[typ]["bump_weight"])
            added.append(typ)
            my_final_dict[typ]['pos_weight']=who_is[typ]["pos_weight"]
            my_final_dict[typ]['ent_weight']=who_is[typ]["ent_weight"]
            my_final_dict[typ]['bump_weight']=who_is[typ]["bump_weight"]
            my_final_dict[typ]['k']=who_is[typ]["k"]
            my_final_dict[typ]['text_weight']=who_is[typ]["text_weight"]

# for typ in whereverb:
#     if whereverb[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', whereverb[typ]["text_weight"])
#             print('pos_weight', whereverb[typ]["pos_weight"])
#             print('ent_weight', whereverb[typ]["ent_weight"])
#             print('k', whereverb[typ]["k"])
#             print('bump_weight', whereverb[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['pos_weight']=whereverb[typ]["pos_weight"]
#             my_final_dict[typ]['ent_weight']=whereverb[typ]["ent_weight"]
#             my_final_dict[typ]['bump_weight']=whereverb[typ]["bump_weight"]
#             my_final_dict[typ]['k']=whereverb[typ]["k"]
#             my_final_dict[typ]['text_weight']=whereverb[typ]["text_weight"]
# for typ in who_adv:
#     if who_adv[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', who_adv[typ]["text_weight"])
#             print('pos_weight', who_adv[typ]["pos_weight"])
#             print('ent_weight', who_adv[typ]["ent_weight"])
#             print('k', who_adv[typ]["k"])
#             print('bump_weight', who_adv[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['pos_weight']=who_adv[typ]["pos_weight"]
#             my_final_dict[typ]['ent_weight']=who_adv[typ]["ent_weight"]
#             my_final_dict[typ]['bump_weight']=who_adv[typ]["bump_weight"]
#             my_final_dict[typ]['k']=who_adv[typ]["k"]
#             my_final_dict[typ]['text_weight']=who_adv[typ]["text_weight"]

# for typ in the_rest:
#     if len(the_rest[typ])>0:
#         if the_rest[typ]["text_weight"]!=0:
#             if typ not in added:
#                 my_final_dict[typ]={}

#                 print('for type ', typ)
#                 print('text_weight', the_rest[typ]["text_weight"])
#                 print('pos_weight', the_rest[typ]["pos_weight"])
#                 print('ent_weight', the_rest[typ]["ent_weight"])
#                 print('k', the_rest[typ]["k"])
#                 print('bump_weight', the_rest[typ]["bump_weight"])
#                 added.append(typ)
#                 my_final_dict[typ]['pos_weight']=the_rest[typ]["pos_weight"]
#                 my_final_dict[typ]['ent_weight']=the_rest[typ]["ent_weight"]
#                 my_final_dict[typ]['bump_weight']=the_rest[typ]["bump_weight"]
#                 my_final_dict[typ]['k']=the_rest[typ]["k"]
#                 my_final_dict[typ]['text_weight']=the_rest[typ]["text_weight"]
#     # print(my_dict)
# # to_add=[]
# print("l")
# for typ in my_dictl:
#     if my_dictl[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', my_dictl[typ]["text_weight"])
#             print('pos_weight', my_dictl[typ]["pos_weight"])
#             print('ent_weight', my_dictl[typ]["ent_weight"])
#             print('k', my_dictl[typ]["k"])
#             print('bump_weight', my_dictl[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['POS']=my_dictl[typ]["pos_weight"]
#             my_final_dict[typ]['ENT']=my_dictl[typ]["ent_weight"]
#             my_final_dict[typ]['BUMP']=my_dictl[typ]["bump_weight"]
#             my_final_dict[typ]['K']=my_dictl[typ]["k"]
#             my_final_dict[typ]['TEXT']=my_dictl[typ]["text_weight"]
# print('s')
# for typ in my_dict:
#     if my_dict[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', my_dict[typ]["text_weight"])
#             print('pos_weight', my_dict[typ]["pos_weight"])
#             print('ent_weight', my_dict[typ]["ent_weight"])
#             print('k', my_dict[typ]["k"])
#             print('bump_weight', my_dict[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['POS']=my_dict[typ]["pos_weight"]
#             my_final_dict[typ]['ENT']=my_dict[typ]["ent_weight"]
#             my_final_dict[typ]['BUMP']=my_dict[typ]["bump_weight"]
#             my_final_dict[typ]['K']=my_dict[typ]["k"]
#             my_final_dict[typ]['TEXT']=my_dict[typ]["text_weight"]

# print('a')
# for typ in my_dicta:
#     if my_dicta[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', my_dicta[typ]["text_weight"])
#             print('pos_weight', my_dicta[typ]["pos_weight"])
#             print('ent_weight', my_dicta[typ]["ent_weight"])
#             print('k', my_dicta[typ]["k"])
#             print('bump_weight', my_dicta[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['POS']=my_dicta[typ]["pos_weight"]
#             my_final_dict[typ]['ENT']=my_dicta[typ]["ent_weight"]
#             my_final_dict[typ]['BUMP']=my_dicta[typ]["bump_weight"]
#             my_final_dict[typ]['K']=my_dicta[typ]["k"]
#             my_final_dict[typ]['TEXT']=my_dicta[typ]["text_weight"]

# print('b')
# for typ in my_dictb:
#     if my_dictb[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', my_dictb[typ]["text_weight"])
#             print('pos_weight', my_dictb[typ]["pos_weight"])
#             print('ent_weight', my_dictb[typ]["ent_weight"])
#             print('k', my_dictb[typ]["k"])
#             print('bump_weight', my_dictb[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['POS']=my_dictb[typ]["pos_weight"]
#             my_final_dict[typ]['ENT']=my_dictb[typ]["ent_weight"]
#             my_final_dict[typ]['BUMP']=my_dictb[typ]["bump_weight"]
#             my_final_dict[typ]['K']=my_dictb[typ]["k"]
#             my_final_dict[typ]['TEXT']=my_dictb[typ]["text_weight"]

# print('c')
# for typ in my_dictc:
#     if my_dictc[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', my_dictc[typ]["text_weight"])
#             print('pos_weight', my_dictc[typ]["pos_weight"])
#             print('ent_weight', my_dictc[typ]["ent_weight"])
#             print('k', my_dictc[typ]["k"])
#             print('bump_weight', my_dictc[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['POS']=my_dictc[typ]["pos_weight"]
#             my_final_dict[typ]['ENT']=my_dictc[typ]["ent_weight"]
#             my_final_dict[typ]['BUMP']=my_dictc[typ]["bump_weight"]
#             my_final_dict[typ]['K']=my_dictc[typ]["k"]
#             my_final_dict[typ]['TEXT']=my_dictc[typ]["text_weight"]
# print('wa')
# for typ in my_dictwa:
#     if my_dictwa[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', my_dictwa[typ]["text_weight"])
#             print('pos_weight', my_dictwa[typ]["pos_weight"])
#             print('ent_weight', my_dictwa[typ]["ent_weight"])
#             print('k', my_dictwa[typ]["k"])
#             print('bump_weight', my_dictwa[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['POS']=my_dictwa[typ]["pos_weight"]
#             my_final_dict[typ]['ENT']=my_dictwa[typ]["ent_weight"]
#             my_final_dict[typ]['BUMP']=my_dictwa[typ]["bump_weight"]
#             my_final_dict[typ]['K']=my_dictwa[typ]["k"]
#             my_final_dict[typ]['TEXT']=my_dictwa[typ]["text_weight"]

# print('r2')
# for typ in my_dictr2:
#     if my_dictr2[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', my_dictr2[typ]["text_weight"])
#             print('pos_weight', my_dictr2[typ]["pos_weight"])
#             print('ent_weight', my_dictr2[typ]["ent_weight"])
#             print('k', my_dictr2[typ]["k"])
#             print('bump_weight', my_dictr2[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['POS']=my_dictr2[typ]["pos_weight"]
#             my_final_dict[typ]['ENT']=my_dictr2[typ]["ent_weight"]
#             my_final_dict[typ]['BUMP']=my_dictr2[typ]["bump_weight"]
#             my_final_dict[typ]['K']=my_dictr2[typ]["k"]
#             my_final_dict[typ]['TEXT']=my_dictr2[typ]["text_weight"]

# print('d')
# for typ in my_dictd:
#     if my_dictd[typ]["text_weight"]!=0:
#         if typ not in added:
#             my_final_dict[typ]={}

#             print('for type ', typ)
#             print('text_weight', my_dictd[typ]["text_weight"])
#             print('pos_weight', my_dictd[typ]["pos_weight"])
#             print('ent_weight', my_dictd[typ]["ent_weight"])
#             print('k', my_dictd[typ]["k"])
#             print('bump_weight', my_dictd[typ]["bump_weight"])
#             added.append(typ)
#             my_final_dict[typ]['POS']=my_dictd[typ]["pos_weight"]
#             my_final_dict[typ]['ENT']=my_dictd[typ]["ent_weight"]
#             my_final_dict[typ]['BUMP']=my_dictd[typ]["bump_weight"]
#             my_final_dict[typ]['K']=my_dictd[typ]["k"]
#             my_final_dict[typ]['TEXT']=my_dictd[typ]["text_weight"]

# to_use_qtype=["where ADP","what ADJ", "how VERB","when PRON","how ADJ", "what VERB", "how AUX", "why VERB","what NUM", "when AUX", "who ADV"]


for typ in my_final_dict:
    print(typ)
    # if(typ not in added):
    #     # if(typ not in to_use_qtype):
    #         print("not added: ",typ)

try: 
    f = open('tuned_weights_MASTER_UPDATED', 'wb') 
    pickle.dump(my_final_dict, f) 
    f.close()
except: 
    print("Something went wrong")


print(my_final_dict)