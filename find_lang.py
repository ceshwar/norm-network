import os
import codecs
import langdetect
from collections import Counter
from multiprocessing import Pool


indir = '/home/samory/preprocessed_data_balanced/'
fnames = [os.path.join(indir, fname) for fname in os.listdir(indir) if fname.endswith('..balanced')]
def wraplangdetect(line):
    try:
        return langdetect.detect(line)
    except:
        return 'unk'
def compute_language_fraction(fname):
    with codecs.open(fname, encoding='utf8', errors='ignore') as f:
        lines = f.readlines()
        lines = [l[l.index(' ')+1:].strip() for l in lines]
        lines = filter(lambda x: x, lines)
        langs = Counter(map(wraplangdetect, lines))
        if 'en' not in langs: return fname, 0.
        return fname, langs['en']/float(sum(langs.values()))
    
if __name__ == "__main__":
    poo = Pool()
    language_ratios = poo.map(compute_language_fraction,fnames)
    with open('language_ratios.txt', 'w+') as f:
        f.write('\n'.join(['%s %2.2f' % (os.path.split( fname)[-1][:-len('..balanced')], lr) for fname, lr in language_ratios]))
