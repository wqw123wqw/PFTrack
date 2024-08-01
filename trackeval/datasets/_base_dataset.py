import csv
import io
import zipfile
import os
import traceback
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from .. import _timing
from ..utils import TrackEvalException


class _BaseDataset(ABC):
    @abstractmethod
    def __init__(self):
        self.tracker_list = None
        self.seq_list = None
        self.class_list = None
        self.output_fol = None
        self.output_sub_fol = None
        self.should_classes_combine = True
        self.use_super_categories = False

    # Functions to implement:

    @staticmethod
    @abstractmethod
    def get_default_dataset_config():
        ...

    @abstractmethod
    def _load_raw_file(self,  seq, is_gt):
        ...

    @_timing.time
    @abstractmethod
    def get_preprocessed_seq_data(self, raw_data, cls):
        ...

    @abstractmethod
    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        ...

    # Helper functions for all datasets:

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def get_name(self):
        return self.get_class_name()

    def get_output_fol(self):
        return self.output_fol

    def get_display_name(self, tracker):
        """ Can be overwritten if the predect_mot name (in files) is different to how it should be displayed.
        By default this method just returns the predect_mot name as is.
        """
        return tracker

    def get_eval_info(self):
        """Return info about the dataset needed for the Evaluator"""
        return self.tracker_list, self.seq_list, self.class_list

    @_timing.time
    def get_raw_seq_data(self,  seq):
        """ Loads raw data (tracker and ground-truth) for a single tracker on a single sequence.
        Raw data includes all of the information needed for both preprocessing and evaluation, for all classes.
        A later function (get_processed_seq_data) will perform such preprocessing and extract relevant information for
        the evaluation of each class.

        This returns a dict which contains the fields:
        [num_timesteps]: integer
        [gt_ids, tracker_ids, gt_classes, tracker_classes, tracker_confidences]:
                                                                list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, tracker_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [similarity_scores]: list (for each timestep) of 2D NDArrays.
        [gt_extras]: dict (for each extra) of lists (for each timestep) of 1D NDArrays (for each det).

        gt_extras contains dataset specific information used for preprocessing such as occlusion and truncation levels.

        Note that similarities are extracted as part of the dataset and not the metric, because almost all metrics are
        independent of the exact method of calculating the similarity. However datasets are not (e.g. segmentation
        masks vs 2D boxes vs 3D boxes).
        We calculate the similarity before preprocessing because often both preprocessing and evaluation require it and
        we don't wish to calculate this twice.
        We calculate similarity between all gt and tracker classes (not just each class individually) to allow for
        calculation of metrics such as class confusion matrices. Typically the impact of this on performance is low.
        """
        # Load raw data.
        raw_gt_data = self._load_raw_file( seq, is_gt=True)  # ['gt_crowd_ignore_regions', 'gt_extras', 'gt_ids', 'gt_classes', 'gt_dets', 'num_timesteps', 'seq']
        raw_tracker_data = self._load_raw_file(seq, is_gt=False)
        raw_data = {**raw_tracker_data, **raw_gt_data}  # Merges dictionaries

        # Calculate similarities for each timestep.
        similarity_scores = []
        for t, (gt_dets_t, tracker_dets_t) in enumerate(zip(raw_data['gt_dets'], raw_data['tracker_dets'])):
            ious = self._calculate_similarities(gt_dets_t, tracker_dets_t)
            similarity_scores.append(ious)
        raw_data['similarity_scores'] = similarity_scores
        return raw_data

    @staticmethod
    def _load_simple_text_file(file, time_col=0, id_col=None, remove_negative_ids=False, valid_filter=None,
                               crowd_ignore_filter=None, convert_filter=None,
                               force_delimiters=None):

        if remove_negative_ids and id_col is None:
            raise TrackEvalException('remove_negative_ids is True, but id_col is not given.')
        if crowd_ignore_filter is None:
            crowd_ignore_filter = {}
        if convert_filter is None:
            convert_filter = {}
        try:
            fp = open(file)
            read_data = {}
            crowd_ignore_data = {}
            fp.seek(0, os.SEEK_END)
            # check if file is empty
            if fp.tell():
                fp.seek(0)
                dialect = csv.Sniffer().sniff(fp.readline(), delimiters=force_delimiters)  # Auto determine structure.
                dialect.skipinitialspace = True  # Deal with extra spaces between columns
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    try:
                        # Deal with extra trailing spaces at the end of rows
                        if row[-1] in '':
                            row = row[:-1]
                        timestep = str(int(float(row[time_col])))
                        # Read ignore regions separately.
                        is_ignored = False
                        for ignore_key, ignore_value in crowd_ignore_filter.items():
                            if row[ignore_key].lower() in ignore_value:
                                # Convert values in one column (e.g. string to id)
                                for convert_key, convert_value in convert_filter.items():
                                    row[convert_key] = convert_value[row[convert_key].lower()]
                                # Save data separated by timestep.
                                if timestep in crowd_ignore_data.keys():
                                    crowd_ignore_data[timestep].append(row)
                                else:
                                    crowd_ignore_data[timestep] = [row]
                                is_ignored = True
                        if is_ignored:  # if det is an ignore region, it cannot be a normal det.
                            continue
                        # Exclude some dets if not valid.
                        if valid_filter is not None:
                            for key, value in valid_filter.items():
                                if row[key].lower() not in value:
                                    continue
                        if remove_negative_ids:
                            if int(float(row[id_col])) < 0:
                                continue
                        # Convert values in one column (e.g. string to id)
                        for convert_key, convert_value in convert_filter.items():
                            row[convert_key] = convert_value[row[convert_key].lower()]
                        # Save data separated by timestep.
                        if timestep in read_data.keys():
                            read_data[timestep].append(row)
                        else:
                            read_data[timestep] = [row]
                    except Exception:
                        exc_str_init = 'In file %s the following line cannot be read correctly: \n' % os.path.basename(
                            file)
                        exc_str = ' '.join([exc_str_init]+row)
                        raise TrackEvalException(exc_str)
            fp.close()
        except Exception:
            print('Error loading file: %s, printing traceback.' % file)
            traceback.print_exc()
            raise TrackEvalException(
                'File %s cannot be read because it is either not present or invalidly formatted' % os.path.basename(
                    file))
        return read_data, crowd_ignore_data

    @staticmethod
    def _calculate_mask_ious(masks1, masks2, is_encoded=False, do_ioa=False):
        """ Calculates the IOU (intersection over union) between two arrays of segmentation masks.
        If is_encoded a run length encoding with pycocotools is assumed as input format, otherwise an input of numpy
        arrays of the shape (num_masks, height, width) is assumed and the encoding is performed.
        If do_ioa (intersection over area) , then calculates the intersection over the area of masks1 - this is commonly
        used to determine if detections are within crowd ignore region.
        :param masks1:  first set of masks (numpy array of shape (num_masks, height, width) if not encoded,
                        else pycocotools rle encoded format)
        :param masks2:  second set of masks (numpy array of shape (num_masks, height, width) if not encoded,
                        else pycocotools rle encoded format)
        :param is_encoded: whether the input is in pycocotools rle encoded format
        :param do_ioa: whether to perform IoA computation
        :return: the IoU/IoA scores
        """

        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils

        # use pycocotools for run length encoding of masks
        if not is_encoded:
            masks1 = mask_utils.encode(np.array(np.transpose(masks1, (1, 2, 0)), order='F'))
            masks2 = mask_utils.encode(np.array(np.transpose(masks2, (1, 2, 0)), order='F'))

        # use pycocotools for iou computation of rle encoded masks
        ious = mask_utils.iou(masks1, masks2, [do_ioa]*len(masks2))
        if len(masks1) == 0 or len(masks2) == 0:
            ious = np.asarray(ious).reshape(len(masks1), len(masks2))
        assert (ious >= 0 - np.finfo('float').eps).all()
        assert (ious <= 1 + np.finfo('float').eps).all()

        return ious

    @staticmethod
    def _calculate_box_ious(bboxes1, bboxes2, box_format='xywh', do_ioa=False):
        """ Calculates the IOU (intersection over union) between two arrays of boxes.
        Allows variable box formats ('xywh' and 'x0y0x1y1').
        If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
        used to determine if detections are within crowd ignore region.
        """
        if box_format in 'xywh':
            # layout: (x0, y0, w, h)
            bboxes1 = deepcopy(bboxes1)
            bboxes2 = deepcopy(bboxes2)

            bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
            bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
            bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
            bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
        elif box_format not in 'x0y0x1y1':
            raise (TrackEvalException('box_format %s is not implemented' % box_format))

        # layout: (x0, y0, x1, y1)
        min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

        if do_ioa:
            ioas = np.zeros_like(intersection)
            valid_mask = area1 > 0 + np.finfo('float').eps
            ioas[valid_mask, :] = intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]

            return ioas
        else:
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
            union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
            intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
            intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
            intersection[union <= 0 + np.finfo('float').eps] = 0
            union[union <= 0 + np.finfo('float').eps] = 1
            ious = intersection / union
            return ious

    @staticmethod
    def _calculate_euclidean_similarity(dets1, dets2, zero_distance=2.0):
        """ Calculates the euclidean distance between two sets of detections, and then converts this into a similarity
        measure with values between 0 and 1 using the following formula: sim = max(0, 1 - dist/zero_distance).
        The default zero_distance of 2.0, corresponds to the default used in MOT15_3D, such that a 0.5 similarity
        threshold corresponds to a 1m distance threshold for TPs.
        """
        dist = np.linalg.norm(dets1[:, np.newaxis]-dets2[np.newaxis, :], axis=2)
        sim = np.maximum(0, 1 - dist/zero_distance)
        return sim

    @staticmethod   # 核对每帧图有唯一的id
    def _check_unique_ids(data, after_preproc=False):
        """Check the requirement that the tracker_ids and gt_ids are unique per timestep"""
        gt_ids = data['gt_ids']
        tracker_ids = data['tracker_ids']
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
            if len(tracker_ids_t) > 0:
                unique_ids, counts = np.unique(tracker_ids_t, return_counts=True)
                if np.max(counts) != 1:
                    duplicate_ids = unique_ids[counts > 1]
                    exc_str_init = 'Tracker predicts the same ID more than once in a single timestep ' \
                                   '(seq: %s, frame: %i, ids:' % (data['seq'], t+1)
                    exc_str = ' '.join([exc_str_init] + [str(d) for d in duplicate_ids]) + ')'
                    if after_preproc:
                        exc_str_init += '\n Note that this error occurred after preprocessing (but not before), ' \
                                        'so ids may not be as in file, and something seems wrong with preproc.'
                    raise TrackEvalException(exc_str)
            if len(gt_ids_t) > 0:
                unique_ids, counts = np.unique(gt_ids_t, return_counts=True)
                if np.max(counts) != 1:
                    duplicate_ids = unique_ids[counts > 1]
                    exc_str_init = 'Ground-truth has the same ID more than once in a single timestep ' \
                                   '(seq: %s, frame: %i, ids:' % (data['seq'], t+1)
                    exc_str = ' '.join([exc_str_init] + [str(d) for d in duplicate_ids]) + ')'
                    if after_preproc:
                        exc_str_init += '\n Note that this error occurred after preprocessing (but not before), ' \
                                        'so ids may not be as in file, and something seems wrong with preproc.'
                    raise TrackEvalException(exc_str)
