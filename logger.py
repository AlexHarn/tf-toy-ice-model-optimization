from __future__ import print_function
import os
import subprocess
import time
from datetime import datetime
from shutil import copyfile


class Logger:
    """
    Initializes the Logger object.

    The logger class is still very basic right now and can only deal with
    homogeneous ice. Plans include the option to track any variables using
    dictionaries and easily define plots. Also TensorBoard integration is a
    crucial planned feature.

    Para
    """
    # ---------------------------- Initialization -----------------------------
    def __init__(self, logdir=None, overwrite=False, log_version=True):
        self._start_time = datetime.utcnow()

        # set default logdir
        if logdir is None:
            # for python 2 support... in python 3 there is datetime.timestamp()
            logdir = './log/{}/'.format(
                int(time.mktime(self._start_time.timetuple())))

        # check if the logdir already exists or create it
        if os.path.exists(logdir):
            if not overwrite:
                raise LogdirAlreadyExistsError("If you want to overwrite the "
                                               "existing log set "
                                               "overwrite=True.")
        else:
            os.makedirs(logdir)

        # copy the currently used settings into the logdir
        copyfile('./settings.py', logdir+'settings.py')

        # set attributes
        self._logdir = logdir

        # start writing into the buffer
        self._buffer = "Starting at " \
            + self._start_time.strftime("%a %b %d %H:%M:%S %Z %Y UTC")+'\n'
        if log_version:
            self._buffer += "Git HEAD points at " \
                + subprocess.check_output(['git', 'rev-parse',
                                           'HEAD']).decode('utf-8') \
                + "Git Status says:\n" \
                + subprocess.check_output(['git', 'status']).decode('utf-8')

    # ----------------------- Writing to Files --------------------------------
    def write(self):
        """
        Writes the current buffers to files.

        Should only be used sparingly to not waist too many resources.
        """
        try:
            with open(self._logdir+'session.log', 'a') as session_logfile:
                session_logfile.write(self._buffer)
            self._buffer = ""
        except Exception:
            print("Logger could not write to file!")
            pass

    # -------------------- Public Logging Methods -----------------------------
    def log(self, step, l_abs_pred, l_scat_pred, printing=True):
        """
        Very basic and specialized logging method, no time left for today but
        this will be greatly improved.

        Parameters
        ----------
        step : integer
            The current training step.
        l_abs_pred : float, length in m
            The current prediction for the absorbtion length.
        l_scat_pred : float, length in m
            The current prediction for the scattering length.
        printing : boolean
            Whether or not to print the logged step.
        """

        session_time = datetime.utcnow() - self._start_time
        hours, remainder = divmod(session_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        line = ("[{:02d}:{:02d}:{:02d} {:08d}] l_abs_pred: {:2.3f} "
                "l_scat_pred: {:2.3f}\n") \
            .format(hours, minutes, seconds, step, l_abs_pred, l_scat_pred)

        if printing:
            print(line[:-1])

        self._buffer += line

    def print(self):
        """
        Prints the current session buffer contents without the last newline.
        """
        print(self._buffer[:-1])


class LogdirAlreadyExistsError(Exception):
    pass
