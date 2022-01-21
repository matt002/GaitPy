import os
from pandas.testing import assert_frame_equal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from gaitpy.signal_features import *
from test_gait import run_gaitpy

def test__signal_features():
    # Set source and destination directories
    src = os.path.abspath(__file__ + '/../../')+'/demo/demo_data.csv'

    # Run gaitpy
    sample_rate = 50  # hertz
    subject_height = 177  # centimeters
    obtained_classify_bouts, obtained_gait_features = run_gaitpy(src, sample_rate, subject_height)

    # Confirm expected results
    expected_classify_bouts = pd.read_hdf(os.path.abspath(__file__ + '/../../')+'/demo/demo_classify_bouts.h5')
    assert_frame_equal(expected_classify_bouts, obtained_classify_bouts)

    expected_gait_features = pd.read_csv(os.path.abspath(__file__ + '/../../')+'/demo/demo_gait_features.csv')
    expected_gait_features['bout_start_time'] = pd.to_datetime(expected_gait_features['bout_start_time'],
                                                               format='%Y-%m-%d %H:%M:%S.%f')
    assert_frame_equal(expected_gait_features, obtained_gait_features)
