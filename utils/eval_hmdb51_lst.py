import json

import numpy as np
import pandas as pd

class HMDBclassification(object):

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 subset='validation', verbose=True, top_k=1):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.verbose = verbose
        self.top_k = top_k
        self.ap = None
        self.hit_at_k = None
        self.ys = {}
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        # if not all([field in data.keys() for field in self.gt_fields]):
            # raise IOError('Please input a valid ground truth file.')

        # Initialize data frame
        activity_index, cidx = {}, 0
        video_lst, label_lst = [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            this_label = v['annotations']['label']
            if this_label not in activity_index:
                activity_index[this_label] = cidx
                self.ys[cidx] = this_label
                cidx += 1
            video_lst.append(videoid)
            label_lst.append(activity_index[this_label])
        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     'label': label_lst})
        ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        # if not all([field in data.keys() for field in self.pred_fields]):
            # raise IOError('Please input a valid prediction file.')

        # Initialize data frame
        video_lst, label_lst, score_lst, begin_lst = [], [], [], []
        for videoid, v in data['results'].items():
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                label_lst.append(label)
                score_lst.append(result['score'])
                begin_lst.append(result['begin'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   'label': label_lst,
                                   'score': score_lst,
                                   'begin': begin_lst})
        return prediction

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        hit_at_k = compute_video_hit_at_k(self.ground_truth,
                                          self.prediction, top_k=self.top_k)
        if self.verbose:
            print('[RESULTS] Performance on HMDB51 untrimmed video '
                   'classification task.')
            print('\tAccuracy@{}: {}'.format(self.top_k, hit_at_k))
            #print '\tAvg Hit@{}: {}'.format(self.top_k, avg_hit_at_k)
        self.hit_at_k = hit_at_k

################################################################################
# Metrics
################################################################################
def compute_video_hit_at_k(ground_truth, prediction, top_k=3):
    """Compute accuracy at k prediction between ground truth and
    predictions data frames. This code is greatly inspired by evaluation
    performed in Karpathy et al. CVPR14.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 'label']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'label', 'score']

    Outputs
    -------
    acc : float
        Top k accuracy score.
    """
    video_ids = np.unique(ground_truth['video-id'].values)
    avg_hits_per_vid = np.zeros(video_ids.size)
    logfile = open("log_frame.txt","w")
    best_label = [12,37,42,45,50]
    for i, vid in enumerate(video_ids):
        pred_idx = prediction['video-id'] == vid
        if not pred_idx.any():
            continue
        this_pred = prediction.loc[pred_idx].reset_index(drop=True)
        pred_label = this_pred['label'].tolist()
        gt_idx = ground_truth['video-id'] == vid
        gt_label = ground_truth.loc[gt_idx]['label'].tolist()
        
        for i, this_label in enumerate(pred_label):
            #print(str(this_label) + ' ' + str(gt_label))
            if this_label in gt_label and this_label in best_label:
                logfile.write(str(this_label) + ' ' + str(vid) + ' ' + str(this_pred['begin'].tolist()[i]) + '\n')
    return 0
