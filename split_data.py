from os import listdir
import json
import random
random.seed(1234)

triples = []

for f in listdir('data/TREx/'):
    rel_id = f.split('.')[0]
    if '_' in rel_id: continue
    obj_dict = {}
    for line in open('data/TREx/' + f).read().split('\n'):
        if line != '':
            l = json.loads(line)
            obj = l['obj_uri']
            if obj not in obj_dict:
                obj_dict[obj] = []
            obj_dict[obj].append(l)
            subj = l['sub_uri']
            rel = l['predicate_id']
            triples.append((subj, rel, obj))

    num_obj = len(obj_dict.keys())
    object_list = list(obj_dict.keys())
    random.shuffle(object_list)
    train_obj = set(object_list[:1 * int(num_obj / 5)])
    test_obj = set(object_list[0 * int(num_obj / 5):])
    print(rel_id, len(train_obj), len(test_obj))
    f_train = open('data/TREx/' + rel_id + '_train.jsonl','w')
    f_test = open('data/TREx/' + rel_id + '_test.jsonl','w')
    for k, v in obj_dict.items():
        if k in train_obj:
            for l in v:
                f_train.write(json.dumps(l) + '\n')
        elif k in test_obj:
            for l in v:
                f_test.write(json.dumps(l) + '\n')
    f_train.close()
    f_test.close()

triples = list(set(triples))
for i in range(len(triples)):
    triples[i] = list(triples[i])
open('LAMA_TREx_triples.json','w').write(json.dumps(triples, indent=4))
