class Gaitpy():
    ''' Gait feature extraction and bout classification from single accelerometer in the lumbar location

    Using vertical acceleration data from the lumbar location, this class includes functions for:
        - Continuous wavelet based method of gait kinematic feature extraction.
        - Machine learning based method of bout classification.

    Parameters
    ----------
    data : str or pandas.core.frame.DataFrame
        Pandas dataframe containing timestamp column and vertical acceleration data during gait, both of type float
        OR
        File path of .csv file containing timestamp column and vertical acceleration data during gait
            - One column should contain unix timestamps of type float -- by default gaitpy will assume the column title is
                                                                         'timestamps' with units in milliseconds
            - A second column should be vertical acceleration of type float -- by default gaitpy will assume the column
                                                                               title is 'y' with units in m/s^2

    sample_rate : int or float
        Sampling rate of accelerometer data in Hertz.

    v_acc_col_name : str
        Column name of the vertical acceleration data ('y' by default)

    ts_col_name : str
        Column name of the timestamps ('timestamps' by default)

    v_acc_units : str
        Units of vertical acceleration data ('m/s^2' by default)
        Options:
            - 'm/s^2' = meters per second squared
            - 'g' = standard gravity

    ts_units : str
        Units of timestamps ('ms' by default)
        Options:
            - 's' = seconds
            - 'ms' = milli-seconds
            - 'us' = microseconds

    flip : bool
        Boolean specifying whether to flip vertical acceleration data before analysis (False by default). Algorithm
        assumes that baseline vertical acceleration data is at -9.8 m/s^2 or -1g. (ie. If baseline data in vertical
        direction is 1g, set 'flip' argument to True)

    down_sample : int or float
        Sampling rate to downsample data to. Helps to standardize results from multiple sensors with different
        sampling rates.
    '''

    def __init__(self, data, sample_rate, v_acc_col_name='y', ts_col_name='timestamps', v_acc_units='m/s^2', ts_units='ms', flip=False, down_sample=50):
        self.data = data
        self.sample_rate = sample_rate
        self.v_acc_col_name = v_acc_col_name
        self.ts_col_name = ts_col_name
        self.v_acc_units = v_acc_units
        self.ts_units = ts_units
        self.flip = flip
        self.down_sample = down_sample

    def extract_features(self, subject_height, subject_height_units='centimeter', sensor_height_ratio=0.53, result_file=None, classified_gait=None, ic_prom=5, fc_prom=25):
        ''' Continuous wavelet transform based method of gait feature detection optimization methods

        Parameters
        ----------
        subject_height : int or float
            Height of the subject. Accepts centimeters by default.

        subject_height_units : str
            Units of provided subject height. Centimeters by default.
            - options: 'centimeter', 'inches', 'meter'

        sensor_height_ratio : float
            Height of the sensor relative to subject height. Calculated: sensor height / subject height

        result_file : str
            Optional argument that accepts .csv filepath string to save resulting gait feature dataframe to.
            None by default. (ie. myfolder/myfile.csv)

        classified_gait : str or pandas.core.frame.DataFrame
            Pandas dataframe containing results of gait bout classification procedure (classify_bouts)
            OR
            File path of .h5 file containing results of gait bout classification procedure (classify_bouts)

        ic_prom : int
            Prominance of initial contact peak detection

        fc_prom : int
            Prominance of final contact peak detection

        '''
        import pandas as pd
        import gaitpy.util as util
        import warnings

        print('\tExtracting features...')

        # Load data
        y_accel, timestamps = util._load_data(self, self.down_sample)

        # If classified gait is provided, load pandas dataframe or h5 file
        if classified_gait is not None:
            if type(classified_gait) is str:
                gait_predictions = pd.read_hdf(classified_gait)
            elif type(classified_gait) is pd.core.frame.DataFrame:
                gait_predictions = classified_gait
            else:
                print('Unable to load classified gait: Please make sure the data is in the correct format, aborting...')
                return
            # Isolate gait bouts
            gait_windows = gait_predictions[gait_predictions['prediction'] == 1]
            if gait_windows.empty:
                print('The classified_gait data indicates no bouts of gait were detected, aborting...')
                return

            # Concatenate concurrent bouts
            gait_bouts = util._concatenate_windows(gait_windows, window_length=3)
        else:
            # if classified_gait is not provided, assume entire timeseries is 1 bout of gait
            start_time = timestamps[0].astype('datetime64[ms]')
            end_time = timestamps.iloc[-1].astype('datetime64[ms]')
            gait_bouts = pd.DataFrame(data={'start_time': [start_time],
                                            'end_time': [end_time],
                                            'bout_length': [(end_time - start_time).item().total_seconds()]})

        all_bout_gait_features = pd.DataFrame()
        bout_n = 1
        # Loop through gait bouts
        for row_n, bout in gait_bouts.iterrows():
            bout_indices = (timestamps.astype('datetime64[ms]') >= bout.start_time) & (timestamps.astype('datetime64[ms]') <= bout.end_time)
            bout_data = pd.DataFrame([])
            bout_data['y'] = pd.DataFrame(y_accel.loc[bout_indices].reset_index(drop=True))
            bout_data['ts'] = timestamps.loc[bout_indices].reset_index(drop=True)
            if len(bout_data.y) < 15:
                warnings.warn('There are too few data points between '+str(bout.start_time)+' and '+str(bout.end_time)+', skipping bout...')
                continue

            # Run CWT Gait Model IC and FC detection
            ic_peaks, fc_peaks = util._cwt(bout_data.y, self.down_sample, ic_prom, fc_prom)

            # Run gait cycle optimization procedure
            pd.options.mode.chained_assignment = None
            optimized_gait = util._optimization(bout_data['ts'], ic_peaks, fc_peaks)
            if optimized_gait.empty or 1 not in list(optimized_gait.Gait_Cycle):
                continue

            # Calculate changes in height of the center of mass
            optimized_gait = util._height_change_com(optimized_gait, bout_data['ts'], bout_data['y'], self.down_sample)

            # Calculate gait features
            sensor_height = util._calculate_sensor_height(subject_height, subject_height_units, sensor_height_ratio)
            gait_features = util._cwt_feature_extraction(optimized_gait, sensor_height)

            # remove center of mass height and gait cycle boolean columns, remove rows with NAs
            gait_features.dropna(inplace=True)
            gait_features.drop(['CoM_height','Gait_Cycle', 'FC_opp_foot'], axis=1, inplace=True)

            gait_features.insert(0, 'bout_number', bout_n)
            gait_features.insert(1, 'bout_length_sec', bout.bout_length)
            gait_features.insert(2, 'bout_start_time', bout.start_time)
            gait_features.insert(5, 'gait_cycles', len(gait_features))
            all_bout_gait_features = all_bout_gait_features.append(gait_features)

            bout_n += 1
        all_bout_gait_features.reset_index(drop=True, inplace=True)
        all_bout_gait_features.iloc[:,7:] = all_bout_gait_features.iloc[:,7:].round(3)

        # Save results
        if result_file:
            try:
                if not result_file.endswith('.csv'):
                    result_file += '.csv'
                all_bout_gait_features.to_csv(result_file, index=False, float_format='%.3f')
            except:
                print('Unable to save data: Please make sure your results directory exists, aborting...')
                return

        if all_bout_gait_features.empty:
            print('\tFeature extraction complete. No gait cycles detected...\n')
        else:
            print('\tFeature extraction complete!\n')

        return all_bout_gait_features

    def plot_contacts(self, gait_features, result_file=None, show_plot=True):
        """ Plot initial and final contacts of lumbar based gait feature extraction

        Parameters
        ----------
        gait_features : pandas.DataFrame or str
            Pandas dataframe containing results of extract_features function
            OR
            File path of .csv file containing results of extract_features function

        result_file : str
            Optional argument that accepts .html filepath string to save resulting gait event plot to.
            None by default. (ie. myfolder/myfile.html)

        show_plot : bool
            Optional boolean argument that specifies whether your plot is displayed. True by default.

        """
        from bokeh.plotting import figure, output_file, save, show
        from bokeh.models import Legend, Span
        import pandas as pd
        import gaitpy.util as util
        import numpy as np

        print('\tPlotting contacts...')

        # Load data
        y_accel, timestamps = util._load_data(self, self.down_sample)
        ts = pd.to_datetime(timestamps, unit='ms')

        # Load gait_features
        try:
            if type(gait_features) is str:
                icfc = pd.read_csv(gait_features)
            elif type(gait_features) is pd.core.frame.DataFrame:
                icfc = gait_features
            else:
                print('Unable to load gait features: Please make sure the gait_features is in the correct format, aborting...')
                return
        except:
            print('Unable to load gait features: Please make sure you have provided the correct filepath or dataframe, aborting...')
            return

        if icfc.empty:
            print('\tGait feature dataframe is empty, aborting...')
            return

        p = figure(plot_width=1200, plot_height=600, x_axis_label='Time', y_axis_label='m/s^2', toolbar_location='above', x_axis_type='datetime')
        # Plot vertical axis
        p1 = p.line(ts, y_accel, line_width=2, line_color='blue')

        # isolate ICs, FCs, and bout start/end times
        minima_time = []
        minima_signal = []
        maxima_time = []
        maxima_signal = []
        bout_starts = []
        bout_ends = []
        ics = pd.to_datetime(icfc.IC, unit='ms')
        fcs = pd.to_datetime(icfc.FC, unit='ms')
        icfc.bout_start_time = icfc.bout_start_time.astype(np.int64).values // 10 ** 6
        bouts = icfc[['bout_number', 'bout_length_sec', 'bout_start_time']].drop_duplicates()
        for ic in ics:
            minima_time.append(ic)
            minima_signal.append(float(y_accel[ts.index[ts == ic]]))
        for fc in fcs:
            maxima_time.append(fc)
            maxima_signal.append(float(y_accel[ts.index[ts == fc]]))
        for row, bout in bouts.iterrows():
            bout_starts.append(bout.bout_start_time)
            bout_ends.append(bout.bout_start_time + (bout.bout_length_sec*1000))

        # add IC and FCs to plot
        p2 = p.circle(minima_time, minima_signal, size=15, color="green", alpha=0.5)
        p3 = p.circle(maxima_time, maxima_signal, size=15, color="darkorange", alpha=0.5)

        # add bout start and end times to plot
        for bout_start in bout_starts:
            start_bout_line = Span(location=bout_start,
                                   dimension='height', line_color='green',
                                   line_dash='solid', line_width=1.5)
            p.add_layout(start_bout_line)
        for bout_end in bout_ends:
            end_bout_line = Span(location=bout_end,
                                 dimension='height', line_color='red',
                                 line_dash='solid', line_width=1.5)
            p.add_layout(end_bout_line)

        # add legend
        legend = Legend(items=[
            ("Acceleration", [p1]),
            ("Initial contact", [p2]),
            ("Final contact", [p3])
        ], location=(10, 300))

        # format plot
        p.add_layout(legend, 'right')
        p.xaxis.axis_label_text_font_size = "16pt"
        p.yaxis.axis_label_text_font_size = "16pt"
        p.axis.major_label_text_font_size = '16pt'
        p.title.align = 'center'
        p.title.text_font_size = '16pt'
        p.xaxis.axis_label_text_font_style = 'normal'
        p.yaxis.axis_label_text_font_style = 'normal'
        p.xaxis.axis_label_standoff = 5
        p.yaxis.axis_label_standoff = 20
        p.legend.label_text_font = "arial"
        p.legend.label_text_font_size = '16pt'
        p.legend.glyph_height = 30

        if show_plot:
            show(p)

        # save plot
        if result_file:
            try:
                if not result_file.endswith('.html'):
                    result_file += '.html'
                output_file(result_file)
                save(p)
            except:
                print('Unable to save data: Please make sure your results directory exists, aborting...')
                return

        print('\tPlot complete!\n')

    def classify_bouts(self, result_file=None):
        """ Gait bout classification using acceleration data in the vertical direction from the lumbar location.

        Parameters
        ----------
        result_file : str
            Optional argument that accepts .h5 filepath string to save resulting predictions to.
            None by default. (ie. myfolder/myfile.h5)

        """
        import pickle
        import pandas as pd
        import os
        import deepdish as dd
        import gaitpy.util as util

        print('\tClassifying bouts of gait...')

        # Load model and feature order
        model_filename = os.path.dirname(os.path.realpath(__file__)) + '/model/model.pkl'
        features_filename = os.path.dirname(os.path.realpath(__file__)) + '/model/feature_order.txt'
        model = pickle.load(open(model_filename, 'rb'))
        feature_order = open(features_filename, 'r').read().splitlines()
        model_sample_rate = 50.

        # Load data and convert to g
        raw_y_accel, ts = util._load_data(self, self.down_sample)
        y_accel = raw_y_accel / 9.80665

        # Resample data if necessary
        if self.down_sample > model_sample_rate:
            data, timestamps = util._resample_data(y_accel, ts, str(1000./model_sample_rate) + 'ms')
        elif self.down_sample == model_sample_rate:
            data = pd.DataFrame({'y': y_accel})
            timestamps = pd.DatetimeIndex(ts.astype('datetime64[ms]'))
        elif self.down_sample < model_sample_rate:
            print('Data sample rate too low for bout detection model. Minimum sample rate required: ' +
                            str(model_sample_rate) + ' hz, aborting...')
            return

        # Extract signal features from vertical acceleration data
        feature_set, start_times_list, end_times_list = util._extract_signal_features(data, timestamps, model_sample_rate)
        feature_set = feature_set[feature_order]

        # Predict
        try:
            pred = model.predict(feature_set)
            predictions_df = pd.DataFrame(
                {'prediction': pred, 'window_start_time': start_times_list, 'window_end_time': end_times_list})
        except:
            print('Unable to make predictions from signal features, aborting...')
            return

        # Save predictions to hdf file
        if result_file:
            try:
                if not result_file.endswith('.h5'):
                    result_file += '.h5'
                predictions_dict = {}
                predictions_dict['predictions'] = predictions_df

                dd.io.save(result_file, predictions_dict)
            except:
                print('Unable to save data: Please make sure your results directory exists, aborting...')
                return

        print('\tBout classification complete!\n')

        return predictions_df
