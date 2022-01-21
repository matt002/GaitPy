import os
import pandas as pd
from pandas.testing import assert_frame_equal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
import time
from gaitpy.gait import *

def run_gaitpy(src, sample_rate, subject_height, dst):
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
    gait_bouts = gaitpy.classify_bouts(result_file=os.path.join(dst,'classify_bouts.h5'))

    #### Extract gait characteristics ####
    gait_features = gaitpy.extract_features(subject_height,
                                            subject_height_units='centimeters',
                                            result_file=os.path.join(dst,'gait_features.csv'),
                                            classified_gait=gait_bouts)

    #### Plot results of gait feature extraction ####
    gaitpy.plot_contacts(gait_features, result_file=os.path.join(dst, 'plot_contacts.html'), show_plot=False)

def run_demo():
    # Set source and destination directories
    src = __file__.split(".py")[0] + "_data.csv"
    dst = input("Please provide a path to a results directory:    ")
    while not os.path.isdir(dst):
        dst = input(
            "\nYour previous entry was not appropriate."
            "\nIt should follow a format similar to /Users/username/Desktop/Results"
            "\nPlease provide a path to a results directory:    "
        )

    # Run gaitpy
    st = time.time()
    try:
        sample_rate = 50  # hertz
        subject_height = 177  # centimeters
        run_gaitpy(src, sample_rate, subject_height, dst)
    except Exception as e:
        print("Error processing: {}\nError: {}".format(src, e))
    stp = time.time()
    print("total run time: {} seconds".format(round(stp-st, 2)))

    # Confirm expected results
    print("Checking extract_features endpoints...")
    expected_gait_features = pd.read_csv(__file__.split(".py")[0] + '_gait_features.csv')
    obtained_gait_features = pd.read_csv(os.path.join(dst, 'gait_features.csv'))
    assert_frame_equal(expected_gait_features, obtained_gait_features)
    print("Checking classify_bouts endpoints...")
    expected_classify_bouts = pd.read_hdf(__file__.split(".py")[0] + '_classify_bouts.h5')
    obtained_classify_bouts = pd.read_hdf(os.path.join(dst, 'classify_bouts.h5'))
    assert_frame_equal(expected_classify_bouts, obtained_classify_bouts)
    print("All tests passed")

if __name__ == "__main__":
    run_demo()
