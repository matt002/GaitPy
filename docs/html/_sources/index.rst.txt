.. GaitPy documentation master file, created by
   sphinx-quickstart on Mon Oct 14 13:09:53 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GaitPy's documentation!
==================================

GaitPy provides python functions to read accelerometry data from a single lumbar-mounted sensor and estimate clinical
characteristics of gait. 

The source code is available on Github: `github.com/matt002/GaitPy <https://github.com/matt002/GaitPy>`_

- Device location: lower back/lumbar
- Sensing modality: Accelerometer
- Sensor data: Vertical acceleration
- Minimum sampling rate: 50Hz

Installation
------------
GaitPy is compatible with python v3.6 on MacOSX, Windows, and Linux.

Installation via pip:

.. code::

   pip install gaitpy

You can also install it from source:

.. code::

   git clone https://github.com/matt002/gaitpy
   cd gaitpy
   python setup.py install

Basic usage
-----------
Gaitpy consists of the following 3 functions:

	1. classify_bouts: If your data consists of gait and non-gait data, run the classify_bouts function to first classify bouts of gait. If your data is solely during gait, this function is not necessary to use.
	2. extract_features: Extract initial contact (IC) and final contact (FC) events from your data and estimate various temporal and spatial gait features.
	3. plot_contacts: Plot the resulting bout detections and IC/FC events alongside your raw accelerometer signal.

Gaitpy accepts a csv file or pandas dataframe that includes a column containing unix timestamps and a column containing
vertical acceleration from a lumbar-mounted sensor. Gaitpy makes three assumptions by default:

	1. Timestamps and vertical acceleration columns are labeled 'timestamps' and 'y' respectively, however this can be changed using the 'v_acc_col_name' and 'ts_col_name' arguments respectively.
	2. Timestamps are in Unix milliseconds and data is in meters per second squared, however this can be be changed using the 'ts_units' and 'v_acc_units' arguments respectively.
	3. Baseline vertical acceleration data is -9.8m/s^2 or -1g. If your baseline data is +9.8m/s^2 or +1g, set the 'flip' argument to True.

Additionally, the sample rate of your device (at least 50Hz) and height of the subject must be provided.

More details about the inputs and ouputs of each of these functions can be found in `Czech et al. 2019 (in preparation) <https://joss.theoj.org/papers/a2233c9e27db0b6625dc56a3f7363875>`_.

.. code::

   from gaitpy.gait import Gaitpy

   raw_data = 'raw-data-path or pandas dataframe'
   sample_rate = 128 # hertz
   subject_height = 170 # centimeters

   #### Create an instance of Gaitpy ####
   gaitpy = Gaitpy(raw_data,                           # Raw data consisting of vertical acceleration from lumbar location and unix timestamps
                   sample_rate,                        # Sample rate of raw data (in Hertz)
                   v_acc_col_name='y',                 # Vertical acceleration column name
                   ts_col_name='timestamps',           # Timestamp column name
                   v_acc_units='m/s^2',                # Units of vertical acceleration
                   ts_units='ms',                      # Units of timestamps
                   flip=False)                         # If baseline data is at +1g or +9.8m/s^2, set flip=True

   #### Classify bouts of gait - Optional (use if your data consists of gait and non-gait periods)####
   gait_bouts = gaitpy.classify_bouts(result_file='/my/folder/classified_gait.h5')     # File to save results to (None by default)

   #### Extract gait characteristics ####
   gait_features = gaitpy.extract_features(subject_height,                               # Subject height
                                           subject_height_units='centimeter',            # Units of subject height
                                           result_file='/my/folder/gait_features.csv',   # File to save results to (None by default)
                                           classified_gait=gait_bouts)                   # Pandas Dataframe or .h5 file results of classify_bouts function (None by default)

   #### Plot results of gait feature extraction ####
   gaitpy.plot_contacts(gait_features,                                     # Pandas Dataframe or .csv file results of extract_features function
                        result_file='/my/folder/plot_contacts.html)',      # File to save results to (None by default)
                        show_plot=True)                                    # Specify whether to display plot upon completion (True by default)

Running the demo
----------------
The demo file provided lets you to test whether GaitPy outputs the expected results on your system.

You may run the demo directly from a terminal window:

.. code::

   cd gaitpy/demo
   python demo.py

You may also run the demo via a python interpreter. In a terminal window start python by typing:

.. code::

   python


In the interpreter window you can then import and run the demo with the following two commands:

.. code::

   from gaitpy.demo import demo
   demo.run_demo()

The demo script will prompt you to type in a results directory. Following the run, results will be saved in the provided
results directory (less than 250kB of data will be saved). Running the demo should take less than a minute, though this
may vary depending on your machine.


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules

Acknowledgements
----------------
The Digital Medicine & Translational Imaging group at Pfizer, Inc supported the development of this package.

License
-------
Gaitpy is under the MIT license



