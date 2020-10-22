all_data_5070 = {
    'mirad':{
        'TP':863014,
        'FP':1930954,
        'FN':533341,
        'TN':118907252},
    'lanid':{
        'TP':1042855,
        'FP':1432094,
        'FN':390655,
        'TN':119919527},
    'unet':{
        'TP' :1271587,
        'FP' :224602,
        'FN' :161923,
        'TN' :121106863},
    'rf':{
        'TP':839556,
        'FP':439679,
        'FN':559871,
        'TN':120783252}}

test_5070 = {
    'mirad':{
        'TP':209996,
        'FP':444353,
        'FN':85087,
        'TN':26642204},
    'lanid':{
        'TP':216813,
        'FP':345007,
        'FN':89530,
        'TN':26865082},
    'unet':{
        'TP':269467,
        'FP':34115,
        'FN':36876,
        'TN':27170992},
    'rf':{
        'TP':241404,
        'FP':79256,
        'FN':54114,
        'TN':27074897}
    }

print('all data')
np =  0
for key, val in all_data_5070.items():
    np = sum(list(val.values()))
    acc = (val['TP'] + val['TN']) / (val['TP'] + val['TN'] + val['FP'] + val['FN'])
    rec = val['TP'] / (val['TP'] + val['FN'])
    prec = val['TP'] / (val['TP'] + val['FP'])
    s = "{}: OA: {:.3f}, precision: {:.3f}, recall: {:.3f}, np: {:.3f}".format(key, acc, prec, rec,
            np)
    print(s)

print('test data')
for key, val in test_5070.items():
    np = sum(list(val.values()))
    acc = (val['TP'] + val['TN']) / (val['TP'] + val['TN'] + val['FP'] + val['FN'])
    rec = val['TP'] / (val['TP'] + val['FN'])
    prec = val['TP'] / (val['TP'] + val['FP'])
    s = "{}: OA: {:.3f}, precision: {:.3f}, recall: {:.3f}, np: {:.3f}".format(key, acc, prec, rec,
            np)
    print(s)
