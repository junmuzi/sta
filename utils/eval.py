from eval_hmdb51_video import *
from eval_ucf101_video import *
from eval_kinetics import *

datasets = 'kinetics'

if datasets == 'hmdb51':
    datachecker = HMDBclassification('/media/lijun_private_datasets/data/hmdb51_1.json', '../results_hmdb51_baseline/test/val.json',top_k=1)
    datachecker.evaluate()
    #datachecker = HMDBclassification('/media/lijun_private_datasets/data/hmdb51_1.json', '../results_hmdb51-64f/test/val.json',top_k=5)
    #datachecker.evaluate()
elif datasets == 'ucf101':
    datachecker = UCFclassification('/media/lijun_private_datasets/data/ucf101_01.json', '../results_ucf101_baseline/test/val.json',top_k=1)
    datachecker.evaluate()
    #datachecker = UCFclassification('/media/lijun_private_datasets/data/ucf101_01.json', '../results_ucf101-64f/test/val.json',top_k=5)
    #datachecker.evaluate()
else:
    datachecker = KINETICSclassification('/root/lijun/kinetics/kinetics.json', '../results_kinetics-64f-rerun/test/val.json',top_k=1)
    datachecker.evaluate()
    datachecker = KINETICSclassification('/root/lijun/kinetics/kinetics.json', '../results_kinetics-64f-rerun/test/val.json',top_k=5)
    datachecker.evaluate()
