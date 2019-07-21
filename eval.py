from eval_hmdb51 import *
from eval_ucf101 import *

datasets = 'hmdb51'

if datasets == 'hmdb51':
    datachecker = HMDBclassification('/media/lijun_private_datasets/data/hmdb51_1.json', '../results_hmdb51/test/val.json',top_k=1)
    datachecker.evaluate()
    datachecker = HMDBclassification('/media/lijun_private_datasets/data/hmdb51_1.json', '../results_hmdb51/test/val.json',top_k=5)
    datachecker.evaluate()
elif datasets == 'ucf101':
    datachecker = UCFclassification('/media/lijun_private_datasets/data/ucf101_01.json', '../results/test/val.json',top_k=1)
    datachecker.evaluate()
    datachecker = UCFclassification('/media/lijun_private_datasets/data/ucf101_01.json', '../results/test/val.json',top_k=5)
    datachecker.evaluate()
