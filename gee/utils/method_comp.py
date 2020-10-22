all_data = {
'mirad':{
'TP':1272317,
'FP':2833875,
'FN':786565,
'TN':175209913},
'unet':{
'TP':1874935,
'FP':330610,
'FN':239557,
'TN':178423010},
'lanid':{
'TP':1538948,
'FP':2124121,
'FN':575544,
'TN':176659230},
'rf':{
'TP':1239647,
'FP':636986,
'FN':823834,
'TN':177957759}}

test = {
'mirad':{'TP':305826,
'FP':652495,
'FN':124862,
'TN':39354529},
'unet':{
'TP':393342,
'FP':50259,
'FN':54300,
'TN':40127563},
'lanid':{
'TP':315890,
'FP':513552,
'FN':131752,
'TN':39671714},
'rf':{
'TP':352685,
'FP':114606,
'FN':78650,
'TN':39988793}
}

print('all data')
mx = 180897843
for key, val in all_data.items():
    np = 100*(mx - sum(list(val.values())))/mx
    acc = (val['TP'] + val['TN']) / (val['TP'] + val['TN'] + val['FP'] + val['FN'])
    rec = val['TP'] / (val['TP'] + val['FN'])
    prec = val['TP'] / (val['TP'] + val['FP'])
    s = "{}: OA: {:.3f}, precision: {:.3f}, recall: {:.3f} ".format(key, acc, prec, rec)
    s = "{}: OA: {:.3f}, precision: {:.3f}, recall: {:.3f}, np: {:.3f}".format(key, acc, prec, rec,
            np)
    print(s)

print('test')
mx = 40632908
for key, val in test.items():
    np = 100*(mx - sum(list(val.values())))/mx
    acc = (val['TP'] + val['TN']) / (val['TP'] + val['TN'] + val['FP'] + val['FN'])
    rec = val['TP'] / (val['TP'] + val['FN'])
    prec = val['TP'] / (val['TP'] + val['FP'])
    s = "{}: OA: {:.3f}, precision: {:.3f}, recall: {:.3f}, np: {:.3f}".format(key, acc, prec, rec,
            np)
    print(s)



#2013
test = {'unet':
            {'TP': 454658,
            'FP': 65510,
            'FN': 52570,
            'TN': 40110887},
        'rf':{
            'TP': 396100,
            'FP': 114597,
            'FN': 93643,
            'TN': 39987377}
        }
np = 10
'''
2013

all_data = {'unet':{
            'TP':2295846,
            'FP':479154,
            'FN':215174,
            'TN':178262391},

           'rf':{
               'TP':1394635,
               'FP':636229,
               'FN':1060525,
               'TN':177946441}
            }
print()
for key, val in test.items():
    acc = (val['TP'] + val['TN']) / (val['TP'] + val['TN'] + val['FP'] + val['FN'])
    rec = val['TP'] / (val['TP'] + val['FN'])
    prec = val['TP'] / (val['TP'] + val['FP'])
    s = "{}: OA: {:.3f}, precision: {:.3f}, recall: {:.3f}, np: {:.3f}".format(key, acc, prec, rec,
            np)
    print(s)

print()
for key, val in all_data.items():
    acc = (val['TP'] + val['TN']) / (val['TP'] + val['TN'] + val['FP'] + val['FN'])
    rec = val['TP'] / (val['TP'] + val['FN'])
    prec = val['TP'] / (val['TP'] + val['FP'])
    s = "{}: OA: {:.3f}, precision: {:.3f}, recall: {:.3f}, np: {:.3f}".format(key, acc, prec, rec,
            np)
    print(s)
2013
'''
