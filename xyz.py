import numpy as np 
import pickle

my_dictl=np.load('./tuned_weights_long', allow_pickle=True)
my_dict=np.load('./tuned_weights_short', allow_pickle=True)
my_dicta=np.load('./tuned_weights_shorta', allow_pickle=True)
my_dictb=np.load('./tuned_weights_shortf_B', allow_pickle=True)
my_dictc=np.load('./tuned_weights_shortf_remaining', allow_pickle=True)
my_dictwa=np.load('./tuned_weights_shortf_whenaux', allow_pickle=True)
my_dictr2=np.load('./tuned_weights_shortf_whenaux', allow_pickle=True)
my_dictd=np.load('./tuned_weights_shortd', allow_pickle=True)




my_final_dict={}


print(my_dict)
# to_add=[]
added=[]
print("l")
for typ in my_dictl:
    if my_dictl[typ]["text_weight"]!=0:
        if typ not in added:
            print('for type ', typ)
            print('text_weight', my_dictl[typ]["text_weight"])
            print('pos_weight', my_dictl[typ]["pos_weight"])
            print('ent_weight', my_dictl[typ]["ent_weight"])
            print('k', my_dictl[typ]["k"])
            print('bump_weight', my_dictl[typ]["bump_weight"])
            added.append(typ)
            my_final_dict[typ]=my_dictl[typ]
print('s')
for typ in my_dict:
    if my_dict[typ]["text_weight"]!=0:
        if typ not in added:
            print('for type ', typ)
            print('text_weight', my_dict[typ]["text_weight"])
            print('pos_weight', my_dict[typ]["pos_weight"])
            print('ent_weight', my_dict[typ]["ent_weight"])
            print('k', my_dict[typ]["k"])
            print('bump_weight', my_dict[typ]["bump_weight"])
            added.append(typ)
            my_final_dict[typ]=my_dict[typ]

print('a')
for typ in my_dicta:
    if my_dicta[typ]["text_weight"]!=0:
        if typ not in added:
            print('for type ', typ)
            print('text_weight', my_dicta[typ]["text_weight"])
            print('pos_weight', my_dicta[typ]["pos_weight"])
            print('ent_weight', my_dicta[typ]["ent_weight"])
            print('k', my_dicta[typ]["k"])
            print('bump_weight', my_dicta[typ]["bump_weight"])
            added.append(typ)
            my_final_dict[typ]=my_dicta[typ]

print('b')
for typ in my_dictb:
    if my_dictb[typ]["text_weight"]!=0:
        if typ not in added:
            print('for type ', typ)
            print('text_weight', my_dictb[typ]["text_weight"])
            print('pos_weight', my_dictb[typ]["pos_weight"])
            print('ent_weight', my_dictb[typ]["ent_weight"])
            print('k', my_dictb[typ]["k"])
            print('bump_weight', my_dictb[typ]["bump_weight"])
            added.append(typ)
            my_final_dict[typ]=my_dictb[typ]

print('c')
for typ in my_dictc:
    if my_dictc[typ]["text_weight"]!=0:
        if typ not in added:
            print('for type ', typ)
            print('text_weight', my_dictc[typ]["text_weight"])
            print('pos_weight', my_dictc[typ]["pos_weight"])
            print('ent_weight', my_dictc[typ]["ent_weight"])
            print('k', my_dictc[typ]["k"])
            print('bump_weight', my_dictc[typ]["bump_weight"])
            added.append(typ)
            my_final_dict[typ]=my_dictc[typ]
print('wa')
for typ in my_dictwa:
    if my_dictwa[typ]["text_weight"]!=0:
        if typ not in added:
            print('for type ', typ)
            print('text_weight', my_dictwa[typ]["text_weight"])
            print('pos_weight', my_dictwa[typ]["pos_weight"])
            print('ent_weight', my_dictwa[typ]["ent_weight"])
            print('k', my_dictwa[typ]["k"])
            print('bump_weight', my_dictwa[typ]["bump_weight"])
            added.append(typ)
            my_final_dict[typ]=my_dictwa[typ]

print('r2')
for typ in my_dictr2:
    if my_dictr2[typ]["text_weight"]!=0:
        if typ not in added:
            print('for type ', typ)
            print('text_weight', my_dictr2[typ]["text_weight"])
            print('pos_weight', my_dictr2[typ]["pos_weight"])
            print('ent_weight', my_dictr2[typ]["ent_weight"])
            print('k', my_dictr2[typ]["k"])
            print('bump_weight', my_dictr2[typ]["bump_weight"])
            added.append(typ)
            my_final_dict[typ]=my_dictr2[typ]
print('d')
for typ in my_dictd:
    if my_dictd[typ]["text_weight"]!=0:
        if typ not in added:
            print('for type ', typ)
            print('text_weight', my_dictd[typ]["text_weight"])
            print('pos_weight', my_dictd[typ]["pos_weight"])
            print('ent_weight', my_dictd[typ]["ent_weight"])
            print('k', my_dictd[typ]["k"])
            print('bump_weight', my_dictd[typ]["bump_weight"])
            added.append(typ)
            my_final_dict[typ]=my_dictd[typ]
# to_use_qtype=["where ADP","what ADJ", "how VERB","when PRON","how ADJ", "what VERB", "how AUX", "why VERB","what NUM", "when AUX", "who ADV"]


for typ in my_dict:
    if(typ not in added):
        # if(typ not in to_use_qtype):
            print("not added: ",typ)

try: 
    f = open('tuned_weights_all', 'wb') 
    pickle.dump(my_final_dict, f) 
    f.close()
except: 
    print("Something went wrong")