import os
import pandas as pd
from pandas.testing import assert_frame_equal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import gaitpy.util as util
from gaitpy.gait import *

def test_load_data():
    # Load/format data
    src = os.path.abspath(__file__ + '/../../')+'/demo/demo_data.csv'
    raw_data = pd.read_csv(src, skiprows=99, names=['timestamps', 'x', 'y', 'z'], usecols=[0, 1, 2, 3])
    raw_data['unix_timestamps'] = pd.to_datetime(raw_data.timestamps, format="%Y-%m-%d %H:%M:%S:%f").values.astype(np.int64) // 10**6
    raw_data = raw_data.iloc[:10,:]

    # Create an instance of GaitPy
    gaitpy = Gaitpy(raw_data,
                    50,
                    v_acc_col_name='y',
                    ts_col_name='unix_timestamps',
                    v_acc_units='g',
                    ts_units='ms',
                    flip=False)

    # Run function being tested
    obtained_y_accel, obtained_ts = util._load_data(gaitpy, gaitpy.down_sample)

    # Confirm expected results
    expected_y_accel = pd.Series(np.array([7.138261,7.177487,7.177487,7.215733,6.177209,7.868856,7.676646,5.792788,4.831736,10.713765]), name='y')
    pd.testing.assert_series_equal(obtained_y_accel, expected_y_accel)

    expected_ts = pd.Series(np.array([1565087150000,1565087150020,1565087150040,1565087150060,1565087150080,
                                      1565087150100,1565087150120,1565087150140,1565087150160,1565087150180]), name='unix_timestamps')
    pd.testing.assert_series_equal(obtained_ts, expected_ts)

def test_extract_signal_features():
    # Load/format data
    src = os.path.abspath(__file__ + '/../../')+'/demo/demo_data.csv'
    raw_data = pd.read_csv(src, skiprows=99, names=['timestamps', 'x', 'y', 'z'], usecols=[0, 1, 2, 3])
    raw_data['unix_timestamps'] = pd.to_datetime(raw_data.timestamps, format="%Y-%m-%d %H:%M:%S:%f").values.astype(np.int64) // 10**6
    data = pd.DataFrame({'y': raw_data.iloc[:150,:].y})
    timestamps = pd.DatetimeIndex(raw_data.iloc[:150,:].unix_timestamps.astype('datetime64[ms]'))

    # Run function being tested
    obtained_feature_set, obtained_start_times_list, obtained_end_times_list = util._extract_signal_features(data, timestamps, 50)

    # Confirm expected results
    expected_feature_set = pd.DataFrame([[2.67261662, 0.04452866, 0.19341188, 0.5859375, 0.28124919, 0.65097933, -12.20492315, 0.59233586, 0.04666667]],
                                        columns=['lumbar_y_bp_filt_[0.5, 3.0]_signal_entropy','lumbar_y_bp_filt_[0.5, 3.0]_rms',
                                                     'lumbar_y_bp_filt_[0.5, 3.0]_range','lumbar_y_bp_filt_[0.5, 3.0]_dom_freq_value',
                                                     'lumbar_y_bp_filt_[0.5, 3.0]_dom_freq_magnitude','lumbar_y_bp_filt_[0.5, 3.0]_dom_freq_ratio',
                                                     'lumbar_y_bp_filt_[0.5, 3.0]_spectral_flatness','lumbar_y_bp_filt_[0.5, 3.0]_spectral_entropy',
                                                     'lumbar_y_bp_filt_[0.5, 3.0]_mean_cross_rate'], index=[0])
    assert_frame_equal(expected_feature_set, obtained_feature_set)

    expected_start_times_list = [pd.Timestamp(year=2019, month=8, day=6, hour=10, minute=25, second=50)]
    assert expected_start_times_list == obtained_start_times_list

    expected_end_times_list = [pd.Timestamp(year=2019, month=8, day=6, hour=10, minute=25, second=52, microsecond=980000)]
    assert expected_end_times_list == obtained_end_times_list

def test_concatenate_bouts():
    # Load/format data
    classify_bouts = pd.read_hdf(os.path.abspath(__file__ + '/../../')+'/demo/demo_classify_bouts.h5')
    gait_windows = classify_bouts[classify_bouts['prediction'] == 1][0:10]

    # Run function being tested
    obtained_gait_bouts = util._concatenate_windows(gait_windows, window_length=3)

    # Confirm expected results
    expected_values = [[3.0, pd.Timestamp(year=2019, month=8, day=6, hour=10, minute=26, second=2, microsecond=500000), pd.Timestamp(year=2019, month=8, day=6, hour=10, minute=25, second=59, microsecond=500000)],
                       [9.0, pd.Timestamp(year=2019, month=8, day=6, hour=10, minute=26, second=17, microsecond=500000), pd.Timestamp(year=2019, month=8, day=6, hour=10, minute=26, second=8, microsecond=500000)],
                       [18.0, pd.Timestamp(year=2019, month=8, day=6, hour=10, minute=26, second=38, microsecond=500000), pd.Timestamp(year=2019, month=8, day=6, hour=10, minute=26, second=20, microsecond=500000)]]
    expected_gait_bouts = pd.DataFrame(expected_values, columns=['bout_length', 'end_time', 'start_time'])
    pd.testing.assert_frame_equal(expected_gait_bouts, obtained_gait_bouts)

def test_cwt():
    # Load/format data
    src = os.path.abspath(__file__ + '/../../')+'/demo/demo_data.csv'
    raw_data = pd.read_csv(src, skiprows=99, names=['timestamps', 'x', 'y', 'z'], usecols=[0, 1, 2, 3])
    raw_data = raw_data.iloc[3500:3700,:]
    raw_data['y'] = raw_data['y'] * 9.80665

    # Run function being tested
    obtained_ic_peaks, obtained_fc_peaks = util._cwt(raw_data.y, 50, 5, 10)

    # Confirm expected results
    np.testing.assert_array_equal(obtained_ic_peaks, [10,  43,  75, 110, 141, 171])
    np.testing.assert_array_equal(obtained_fc_peaks, [19,  50,  84, 117, 148, 179])

def test_optimization():
    # Load/format data
    src = os.path.abspath(__file__ + '/../../')+'/demo/demo_data.csv'
    raw_data = pd.read_csv(src, skiprows=99, names=['timestamps', 'x', 'y', 'z'], usecols=[0, 1, 2, 3])
    raw_data['unix_timestamps'] = pd.to_datetime(raw_data.timestamps, format="%Y-%m-%d %H:%M:%S:%f").values.astype(np.int64) // 10**6
    raw_data = raw_data.iloc[:500,:]
    ic_peaks = np.array([10,  43,  75, 110, 141, 171])
    fc_peaks = np.array([19,  50,  84, 117, 148, 179])

    # Run function being tested
    obtained_optimization = util._optimization(raw_data['unix_timestamps'], ic_peaks, fc_peaks)

    # Confirm expected results
    expected_optimization = pd.DataFrame([[1565087150200, 1565087151000, 1565087150380, np.nan, 1],
                                          [1565087150860, 1565087151680, 1565087151000, np.nan, 1],
                                          [1565087151500, 1565087152340, 1565087151680, np.nan, 1],
                                          [1565087152200, 1565087152960, 1565087152340, np.nan, 0],
                                          [1565087152820, 1565087153580, 1565087152960, np.nan, 0]],
                                         columns=['IC','FC','FC_opp_foot','CoM_height','Gait_Cycle'])

    pd.testing.assert_frame_equal(expected_optimization, obtained_optimization)

def test_height_change_com():
    # Load/format data
    src = os.path.abspath(__file__ + '/../../')+'/demo/demo_data.csv'
    raw_data = pd.read_csv(src, skiprows=99, names=['timestamps', 'x', 'y', 'z'], usecols=[0, 1, 2, 3])
    raw_data['unix_timestamps'] = pd.to_datetime(raw_data.timestamps, format="%Y-%m-%d %H:%M:%S:%f").values.astype(np.int64) // 10**6
    raw_data = raw_data.iloc[:500,:]
    optimization = pd.DataFrame([[1565087150200, 1565087151000, 1565087150380, np.nan, 1],
                                 [1565087150860, 1565087151680, 1565087151000, np.nan, 1],
                                 [1565087151500, 1565087152340, 1565087151680, np.nan, 1],
                                 [1565087152200, 1565087152960, 1565087152340, np.nan, 0],
                                 [1565087152820, 1565087153580, 1565087152960, np.nan, 0]],
                                columns=['IC','FC','FC_opp_foot','CoM_height','Gait_Cycle'])

    # Run function being tested
    obtained_height_change_com = util._height_change_com(optimization, raw_data['unix_timestamps'], raw_data['y'], 50)
    obtained_height_change_com['CoM_height'] = obtained_height_change_com.CoM_height.round(6)

    # Confirm expected results
    expected_height_change_com = pd.DataFrame([[1565087150200, 1565087151000, 1565087150380, 0.001516, 1],
                                               [1565087150860, 1565087151680, 1565087151000, 0.001538, 1],
                                               [1565087151500, 1565087152340, 1565087151680, 0.001385, 1],
                                               [1565087152200, 1565087152960, 1565087152340, 0.000149, 0],
                                               [1565087152820, 1565087153580, 1565087152960, np.nan, 0]],
                                              columns=['IC','FC','FC_opp_foot','CoM_height','Gait_Cycle'])
    pd.testing.assert_frame_equal(expected_height_change_com, obtained_height_change_com)

def test_calculate_sensor_height():
    # Run function being tested
    obtained_sensor_height = util._calculate_sensor_height(177, 'centimeters', 0.53)

    # Confirm expected results
    assert obtained_sensor_height == 0.9381

def test_cwt_feature_extraction():
    # Load/format data
    optimized_gait = pd.DataFrame([[1565087150200, 1565087151000, 1565087150380, 0.001516, 1],
                                   [1565087150860, 1565087151680, 1565087151000, 0.001538, 1],
                                   [1565087151500, 1565087152340, 1565087151680, 0.001385, 1],
                                   [1565087152200, 1565087152960, 1565087152340, 0.000149, 0],
                                   [1565087152820, 1565087153580, 1565087152960, np.nan, 0]],
                                  columns=['IC','FC','FC_opp_foot','CoM_height','Gait_Cycle'])

    # Run function being tested
    obtained_cwt_feature_extraction = util._cwt_feature_extraction(optimized_gait, 0.9831).round(5)

    # Confirm expected results
    expected_cwt_feature_extraction = pd.DataFrame([[1565087150200, 1565087151000, 1565087150380, 0.00152, 1, 5, 1.30000, 0.04000, 0.66000, 0.02000, 90.90909, 0.18000, 0.04000, 0.14000, 0.04000, 0.32000, 0.00000, np.nan, np.nan, 0.80000, 0.02000, 0.50000, 0.02000, 0.10915, 0.00079, 0.21909, 0.00482, 0.16853],
                                   [1565087150860, 1565087151680, 1565087151000, 0.00154, 1, 5, 1.34000, 0.02000, 0.64000, 0.06000, 93.75000, 0.14000, 0.04000, 0.18000, 0.04000, 0.32000, 0.00000, 0.50000, 0.02000, 0.82000, 0.02000, 0.52000, 0.04000, 0.10994, 0.00561, 0.21427, np.nan, 0.15990],
                                   [1565087151500, 1565087152340, 1565087151680, 0.00138, 1, 5, 1.32000, np.nan, 0.70000, np.nan, 85.71429, 0.18000, np.nan, 0.14000, np.nan, 0.32000, np.nan, 0.52000, np.nan, 0.84000, np.nan, 0.48000, np.nan, 0.10433, np.nan, np.nan, np.nan, np.nan],
                                   [1565087152200, 1565087152960, 1565087152340, 0.00015, 0, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                   [1565087152820, 1565087153580, 1565087152960, np.nan, 0, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
                                  columns=['IC','FC','FC_opp_foot','CoM_height','Gait_Cycle','steps','stride_duration','stride_duration_asymmetry','step_duration','step_duration_asymmetry','cadence','initial_double_support','initial_double_support_asymmetry','terminal_double_support','terminal_double_support_asymmetry','double_support','double_support_asymmetry','single_limb_support','single_limb_support_asymmetry','stance','stance_asymmetry','swing','swing_asymmetry','step_length','step_length_asymmetry','stride_length','stride_length_asymmetry','gait_speed'])
    pd.testing.assert_frame_equal(expected_cwt_feature_extraction, obtained_cwt_feature_extraction)
