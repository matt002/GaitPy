def _butter_lowpass(cutoff, fs, order):
    from scipy.signal import butter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def _butter_lowpass_filter(data, fs, cutoff=20, order=4):
    from scipy import signal
    b, a = _butter_lowpass(cutoff, fs, order=order)
    filtered_signal = signal.filtfilt(b, a, data)
    return filtered_signal

def _band_pass_filter(data_df, sampling_rate, bp_cutoff, order, channels=['X', 'Y', 'Z']):
    from scipy import signal
    import pandas as pd
    import warnings
    import numpy as np

    data = data_df[channels].values

    # Calculate the critical frequency (radians/sample) based on cutoff frequency (Hz) and sampling rate (Hz)
    critical_frequency = [bp_cutoff[0]* 2.0 / sampling_rate, bp_cutoff[1]* 2.0 / sampling_rate]

    # Get the numerator (b) and denominator (a) of the IIR filter
    [b, a] = signal.butter(N=order, Wn=critical_frequency, btype='bandpass', analog=False)

    # Apply filter to raw data
    if np.isnan(np.sum(data)):
        data = np.nan_to_num(data)
        warnings.warn('There are NaN values in your acceleration data. Converting them to 0...')
    bp_filtered_data = signal.filtfilt(b, a, data, padlen=10, axis=0)

    new_channel_labels = [ax + '_bp_filt_' + str(bp_cutoff) for ax in channels]

    data_df[new_channel_labels] = pd.DataFrame(bp_filtered_data)

    return data_df


def _detect_peaks(y, prominence):
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(y, prominence=prominence)

    return peaks

def _cwt(y_accel, sample_rate, ic_prom, fc_prom):
    from scipy import signal, integrate
    import pywt
    import pandas as pd

    # CWT wavelet scale parameters
    scale_cwt1 = float(sample_rate) / 5.0
    scale_cwt2 = float(sample_rate) / 6.0

    # Detrend data
    detrended_data = signal.detrend(y_accel)

    # Low pass filter if less than 40 hz
    if sample_rate >= 40:
        filtered_data = _butter_lowpass_filter(detrended_data, sample_rate)
    elif sample_rate < 40:
        filtered_data = detrended_data

    # cumulative trapezoidal integration
    integrated_data = integrate.cumtrapz(-filtered_data)

    # Gaussian continuous wavelet transform
    cwt_1, freqs = pywt.cwt(integrated_data, scale_cwt1, 'gaus1')
    differentiated_data = cwt_1[0]

    # initial contact (heel strike) peak detection
    ic_peaks = _detect_peaks(pd.Series(-differentiated_data), ic_prom)

    # Gaussian continuous wavelet transform
    cwt_2, freqs = pywt.cwt(-differentiated_data, scale_cwt2, 'gaus1')
    re_differentiated_data = cwt_2[0]

    # final contact (toe off) peak detection
    fc_peaks = _detect_peaks(pd.Series(re_differentiated_data), fc_prom)

    return ic_peaks, fc_peaks

def _calculate_sensor_height(subject_height, units, sensor_height_ratio):
    import warnings
    # calculate lumbar sensor height in meters
    if type(subject_height) is int or type(subject_height) is float:
        if units == 'inches' or units == 'inch' or units == 'in':
            subj_height = subject_height * 0.0254
        elif units == 'centimeters' or units == 'centimeter' or units == 'cm':
            subj_height = subject_height * 0.01
        elif units == 'meters' or units == 'meter' or units == 'm':
            subj_height = subject_height
        else:
            raise ValueError('Unable to identify units for subject height: Please make sure subject_height_units is correctly set.')

        sensor_height = subj_height * sensor_height_ratio
    else:
        warnings.warn('Subject height must be provided as an integer or float. Without subject height, gaitpy will not be able to calculate spatial features.')
        sensor_height = None

    return sensor_height

def _cwt_feature_extraction(cwt_features, sensor_height):
    import numpy as np

    # Calculate CWT model features
    # steps
    cwt_features['steps'] = len(cwt_features)

    # gait cycle duration
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'stride_duration'] = (cwt_features.IC.shift(-2) - cwt_features.IC) / 1000
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'stride_duration_asymmetry'] = abs(cwt_features.stride_duration.shift(-1) - cwt_features.stride_duration)

    # step time
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'step_duration'] = (cwt_features.IC.shift(-1) - cwt_features.IC) / 1000
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'step_duration_asymmetry'] = abs(cwt_features.step_duration.shift(-1) - cwt_features.step_duration)

    # cadence
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'cadence'] = 60 / cwt_features.step_duration

    # initial double support
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'initial_double_support'] = (cwt_features.FC_opp_foot - cwt_features.IC) / 1000
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'initial_double_support_asymmetry'] = abs(cwt_features.initial_double_support.shift(-1) - cwt_features.initial_double_support)

    # terminal double support
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'terminal_double_support'] = (cwt_features.FC - cwt_features.IC.shift(-1)) / 1000
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'terminal_double_support_asymmetry'] = abs(cwt_features.terminal_double_support.shift(-1) - cwt_features.terminal_double_support)

    # double support
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'double_support'] = cwt_features.initial_double_support + cwt_features.terminal_double_support
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'double_support_asymmetry'] = abs(cwt_features.double_support.shift(-1) - cwt_features.double_support)

    # single limb support
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'single_limb_support'] = (cwt_features.IC.shift(-1) - cwt_features.FC.shift(1)) / 1000
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'single_limb_support_asymmetry'] = abs(cwt_features.single_limb_support.shift(-1) - cwt_features.single_limb_support)

    # stance
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'stance'] = (cwt_features.FC - cwt_features.IC) / 1000
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'stance_asymmetry'] = abs(cwt_features.stance.shift(-1) - cwt_features.stance)

    # swing
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'swing'] = (cwt_features.IC.shift(-2) - cwt_features.FC) / 1000
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'swing_asymmetry'] = abs(cwt_features.swing.shift(-1) - cwt_features.swing)

    # step length
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'step_length'] = 2*(np.sqrt((2*sensor_height*cwt_features.CoM_height)-(cwt_features.CoM_height**2)))
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'step_length_asymmetry'] = abs(cwt_features.step_length.shift(-1) - cwt_features.step_length)

    # stride length
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'stride_length'] = cwt_features.step_length.shift(-1) + cwt_features.step_length
    # asymmetry
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'stride_length_asymmetry'] = abs(cwt_features.stride_length.shift(-1) - cwt_features.stride_length)

    # step velocity
    cwt_features.loc[cwt_features['Gait_Cycle'] == 1, 'gait_speed'] = cwt_features.stride_length / cwt_features.stride_duration

    return(cwt_features)

def _optimization(timestamps, ic_peaks, fc_peaks):
    import pandas as pd
    import numpy as np

    # Parameters
    gait_cycle_forward_ic = 2.25  # maximum allowable time (seconds) between initial contact of same foot
    loading_forward_fc = gait_cycle_forward_ic * 0.2  # maximum time (seconds) for loading phase
    stance_forward_fc = (gait_cycle_forward_ic / 2) + loading_forward_fc  # maximum time (seconds) for stance phase

    # Optimization 1: ---Loading Response--- Each IC requires 1 forward FC within 0.225 seconds (opposite foot toe off)
    #              2: ---Stance Phase--- Each IC requires atleast 2 forward FC's within 1.35 second (2nd FC is current IC's matching FC)
    ic_times = timestamps.iloc[ic_peaks]
    fc_times = timestamps.iloc[fc_peaks]

    optimized_gait = pd.DataFrame([])
    for i in range(0, len(ic_times)):
        current_ic = ic_times.iloc[i]
        loading_forward_max = current_ic + (loading_forward_fc * 1000.)
        stance_forward_max = current_ic + (stance_forward_fc * 1000.)

        loading_forward_fcs = fc_times[(fc_times > current_ic) & (fc_times < loading_forward_max)]
        stance_forward_fcs = fc_times[(fc_times > current_ic) & (fc_times < stance_forward_max)]

        if len(loading_forward_fcs) == 1 and len(stance_forward_fcs) >= 2:
            icfc = pd.DataFrame({'IC': [current_ic], 'FC': [stance_forward_fcs.iloc[1]], 'FC_opp_foot': [stance_forward_fcs.iloc[0]]},
                                columns=['IC', 'FC', 'FC_opp_foot'])
            optimized_gait = optimized_gait.append(icfc)

    optimized_gait = optimized_gait.reset_index(drop=True)
    optimized_gait['CoM_height'] = np.nan
    optimized_gait['Gait_Cycle'] = 0

    # Optimization 3: ---Gait Cycles--- Each ic requires atleast 2 ics within 2.25 seconds after
    for i in range(0, len(optimized_gait) - 2):
        current_ic = optimized_gait.IC.iloc[i] / 1000.
        post_ic = optimized_gait.IC.iloc[i + 1] / 1000.
        post_2_ic = optimized_gait.IC.iloc[i + 2] / 1000.

        interval_1 = abs(post_ic - current_ic)
        interval_2 = abs(post_2_ic - current_ic)

        if interval_1 <= gait_cycle_forward_ic and interval_2 <= gait_cycle_forward_ic:
            optimized_gait.Gait_Cycle.iloc[i] = 1

    return optimized_gait

def _height_change_com(optimized_gait, timestamps, gait_data, sample_rate):
    from scipy import signal, integrate
    # Changes in Height of the Center of Mass
    for i in range(0, len(optimized_gait) - 1):
        ic_index = timestamps.index[timestamps == optimized_gait.IC[i]].item()
        post_ic_index = timestamps.index[timestamps == optimized_gait.IC[i + 1]].item()

        step_raw = gait_data.loc[ic_index:post_ic_index]
        if len(step_raw) <= 15:
            continue
        step_detrended = signal.detrend(step_raw)
        if sample_rate >= 40:
            step_filtered = _butter_lowpass_filter(step_detrended, sample_rate)
        elif sample_rate < 40:
            step_filtered = step_detrended
        step_integrate_1 = integrate.cumtrapz(step_filtered) / sample_rate
        step_integrate_2 = integrate.cumtrapz(step_integrate_1) / sample_rate

        h = max(step_integrate_2) - min(step_integrate_2)
        optimized_gait.CoM_height.iloc[i] = h

    return optimized_gait

def _resample_data(y_accel, timestamps, new_fs):
    import pandas as pd

    # Concatenate data and timestamps
    data = pd.DataFrame({'ts': timestamps.astype('datetime64[ms]'), 'y': y_accel})
    data.set_index('ts', inplace=True)

    # Resample data
    resampled_data = pd.DataFrame(data['y'].resample(new_fs).fillna('nearest'))

    # Create resampled timestamp dataframe
    resampled_timestamps = resampled_data.index

    # Reset index of resampled data
    resampled_data.reset_index(drop=True, inplace=True)

    return resampled_data, resampled_timestamps

def _load_data(self, down_sample):
    import pandas as pd
    import numpy as np

    # Load data
    try:
        if type(self.data) is str:
            data_df = pd.read_csv(self.data)
        elif type(self.data) is pd.core.frame.DataFrame:
            data_df = self.data
        else:
            raise Exception('Unable to load data: Please make sure the data is in the correct format.')
    except:
        raise Exception('Unable to load data: Please make sure you have provided the correct filepath.')

    # Check for NaN
    if data_df[self.v_acc_col_name].isnull().any():
        raise Exception('Unable to load data: Please remove all NaN values from your data.')

    # Convert timestamps to milliseconds. Convert data to m/s^2. Flip data if specified.
    try:
        if self.ts_units == 's':
            timestamps = data_df[self.ts_col_name] * 1000.
        elif self.ts_units == 'ms':
            timestamps = data_df[self.ts_col_name]
        elif self.ts_units == 'us':
            timestamps = data_df[self.ts_col_name] / 1000.

        if self.v_acc_units == 'm/s^2':
            y_accel = data_df[self.v_acc_col_name]
        elif self.v_acc_units == 'g':
            y_accel = data_df[self.v_acc_col_name] * 9.80665
        else:
            raise Exception("Unable to load data: Please make sure the units you provide are either 'm/s^2' or 'g'")

        if self.flip == True:
            y_accel = y_accel * -1
    except:
        raise Exception('Unable to load data: Please make sure your columns are labeled correctly.')

    # Downsample data
    if self.sample_rate > down_sample:
        resampled_data, resampled_ts = _resample_data(y_accel, timestamps, str(1000. / down_sample) + 'ms')
        data = pd.Series(resampled_data.y)
        ts = pd.Series(resampled_ts.astype(np.int64).values // 10**6)
    elif self.sample_rate == down_sample:
        data = y_accel
        ts = timestamps
    elif self.sample_rate < down_sample:
        print('Data sample rate too low for extract_feature analysis. Minimum sample rate required: ' +
              str(down_sample) + ' hz, aborting...')
        return

    return data, ts

def _extract_signal_features(data, timestamps, sample_rate, window_length=3.0):
    import pandas as pd
    from . import signal_features as sf
    import warnings

    data.reset_index(drop=True, inplace=True)
    filtered_data_df = _band_pass_filter(data, sample_rate, [0.5, 3.0], 1, channels=['y'])
    total_data_channels = ['y_bp_filt_[0.5, 3.0]']

    # Initialize final DataFrame
    location_feature_and_label_set = pd.DataFrame()

    # Segment into 3 second windows
    total_samples = filtered_data_df.shape[0]
    window_samples = sample_rate * window_length
    total_windows = round(total_samples / float(window_samples))

    start_times_list = []
    end_times_list = []

    for win in range(int(total_windows)):
        # Isolate data into windows
        current_win_start = int(window_samples * win)
        current_win_end = int(current_win_start + window_samples)
        if current_win_end >= filtered_data_df.shape[0]:
            window_data_df = filtered_data_df.loc[current_win_start:, :]
            end_time = timestamps[-1]
        else:
            window_data_df = filtered_data_df.loc[current_win_start:current_win_end, :]
            end_time = timestamps[current_win_end]
        start_time = timestamps[current_win_start]
        window_data_df.reset_index(drop=True, inplace=True)

        # Extract Features
        try:
            features_df = sf._signal_features(window_data_df, total_data_channels, sample_rate)
        except:
            warnings.warn('Error calculating signal features for 3-second window between '+str(current_win_start)+' and '+str(current_win_end)+', skipping window...')
            continue

        # Discard window if NaN's in feature matrix
        if features_df.isnull().values.any():
            warnings.warn('Error calculating signal features for 3-second window between '+str(current_win_start)+' and '+str(current_win_end)+', skipping window...')
            continue

        # Rename columns with device location
        location_columns = ['lumbar_' + s for s in features_df.columns]
        features_df.columns = location_columns

        # Aggregate features for each window
        location_feature_and_label_set = location_feature_and_label_set.append(features_df, ignore_index=True)
        start_times_list.append(start_time)
        end_times_list.append(end_time)

    return location_feature_and_label_set, start_times_list, end_times_list

def _concatenate_windows(windows, window_length):
    import pandas as pd

    concatenated_windows_df = pd.DataFrame([])
    windows = windows.reset_index(drop=True)
    window = 0
    while window < len(windows):
        # concatenate windows of gait from bout windows
        start_time = windows.window_start_time[window]
        window_boolean = True
        while window_boolean:
            end_time = windows.window_end_time[window]
            if window == (len(windows) - 1):
                window_boolean = False
            elif (windows.window_start_time[window+1] - windows.window_start_time[window]).total_seconds() == float(window_length):
                window = window + 1
            else:
                window_boolean = False
        window = window + 1

        # append concatenated windows start and end time to dataframe
        start_end_df = pd.DataFrame(data={'start_time': [start_time],
                                          'end_time': [end_time],
                                          'bout_length': [(end_time - start_time).total_seconds()]})
        concatenated_windows_df = concatenated_windows_df.append(start_end_df)
        concatenated_windows_df.reset_index(drop=True, inplace=True)

    return concatenated_windows_df
