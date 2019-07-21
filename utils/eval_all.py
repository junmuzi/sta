from eval_hmdb51 import *
from eval_ucf101 import *
import sys


def main(path, datasets, number):
    if datasets == 'hmdb51':
        datachecker = HMDBclassification('/media/lijun_private_datasets/data/hmdb51_' + number + '.json', '../' + path + '/test/val.json',top_k=1)
        datachecker.evaluate()
        datachecker = HMDBclassification('/media/lijun_private_datasets/data/hmdb51_' + number + '.json', '../' + path + '/test/val.json',top_k=5)
        datachecker.evaluate()
    elif datasets == 'ucf101':
        datachecker = UCFclassification('/media/lijun_private_datasets/data/ucf101_0' + number + '.json', '../' + path + '/test/val.json',top_k=1)
        datachecker.evaluate()
        datachecker = UCFclassification('/media/lijun_private_datasets/data/ucf101_0' + number + '.json', '../' + path + '/test/val.json',top_k=5)
        datachecker.evaluate()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
