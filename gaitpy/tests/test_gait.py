import os
import pandas as pd
from pandas.testing import assert_frame_equal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from gaitpy.gait import *

def run_gaitpy(src, sample_rate, subject_height):
    # Load/format data
    raw_data = pd.read_csv(src, skiprows=99, names=['timestamps', 'x', 'y', 'z'], usecols=[0, 1, 2, 3])
    raw_data['unix_timestamps'] = pd.to_datetime(raw_data.timestamps, format="%Y-%m-%d %H:%M:%S:%f").values.astype(np.int64) // 10**6

    ### Create an instance of GaitPy ###
    gaitpy = Gaitpy(raw_data,
                    sample_rate,
                    v_acc_col_name='y',
                    ts_col_name='unix_timestamps',
                    v_acc_units='g',
                    ts_units='ms',
                    flip=False)

    #### Classify bouts of gait ####
    gait_bouts = gaitpy.classify_bouts()

    #### Extract gait characteristics ####
    gait_features = gaitpy.extract_features(subject_height,
                                            subject_height_units='centimeters',
                                            classified_gait=gait_bouts)
    return gait_bouts, gait_features

def test_gaitpy():
    # Set source and destination directories
    src = os.path.abspath(__file__ + '/../../')+'/demo/demo_data.csv'

    # Run gaitpy
    obtained_classify_bouts, obtained_gait_features = run_gaitpy(src, 50, 177)

    # Confirm expected results
    expected_classify_bouts = pd.read_hdf(os.path.abspath(__file__ + '/../../')+'/demo/demo_classify_bouts.h5')
    assert_frame_equal(expected_classify_bouts, obtained_classify_bouts)

    expected_gait_features = pd.read_csv(os.path.abspath(__file__ + '/../../')+'/demo/demo_gait_features.csv')
    expected_gait_features['bout_start_time'] = pd.to_datetime(expected_gait_features['bout_start_time'],
                                                               format='%Y-%m-%d %H:%M:%S.%f')
    assert_frame_equal(expected_gait_features, obtained_gait_features)
