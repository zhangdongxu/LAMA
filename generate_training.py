import json

test_data = {}
for d in json.load(open('LAMA_TREx_triples.json')):
    sid, p, oid = d[0], d[1], d[2]
    if p not in test_data: test_data[p] = set()
    test_data[p].add(sid + '::::' + oid)

all_data = json.load(open('all_TREx_triples.json'))

train_data = {}
for p, epid_ep_freq in all_data.items():
    if p in test_data:
        for epid, ep_freq in epid_ep_freq.items():
            if epid not in test_data[p]:
                for ep, freq in ep_freq.items():
                    if freq > 0:
                        if p not in train_data: train_data[p] = {}
                        if epid not in train_data[p]: train_data[p][epid] = set()
                        train_data[p][epid].add(ep)


for p, epid_eps in train_data.items():
    f = open('data/TREx/' + p + '_train.jsonl','w')
    for epid, eps in epid_eps.items():
        sid, oid = epid.split('::::')
        json_line = {"uuid":"", "obj_uri":oid, "obj_label":"", "sub_uri":sid, "sub_label":"", "predicate_id": p, "evidences":[]}
        subjs = []
        objs = []
        for ep in list(eps):
            s, o = ep.split('::::')
            subjs.append(s)
            objs.append(o)
            json_line['evidences'].append({"sub_surface":s, "obj_surface":o, "masked_sentence":""})
        subj = sorted(subjs, key=lambda x:len(x))[-1]
        obj = sorted(objs, key=lambda x:len(x))[-1]
        json_line['obj_label'] = obj
        json_line['sub_label'] = subj
        f.write(json.dumps(json_line) + '\n')
    f.close()
