from __future__ import print_function
import os
import subprocess
import time
import pandas as pd
from datetime import datetime
from shutil import copyfile


class Logger:
    """
    Initializes the Logger object.

    The logger class is still very basic right now and can only deal with
    homogeneous ice.

    Parameters
    ----------
    logdir : string
        Path to the logdir to write to. If None a directory inside ./log gets
        created as logdir. Its name is the current timestamp.
    overwrite : boolean
        If True the logdir will get overwritten if it already exists. Caution
        is advised, it is always recommended to set this parameter to False in
        which case an exception is raised if the logdir already exists.

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

            # clean the existing logdir if overwrite is True
            if os.path.isfile(logdir+'session.log'):
                os.unlink(logdir+'session.log')
            if os.path.isfile(logdir+'variables.hdf5'):
                os.unlink(logdir+'variables.hdf5')
        else:
            os.makedirs(logdir)

        # copy the currently used settings into the logdir
        copyfile('./settings.py', logdir+'settings.py')

        # create version logfile
        if log_version:
            version_log = "Git HEAD points at " \
                + subprocess.check_output(['git', 'rev-parse',
                                           'HEAD']).decode('utf-8') \
                + "Git Status says:\n" \
                + subprocess.check_output(['git', 'status']).decode('utf-8')
            try:
                with open(logdir+'version.log', 'w') as version_logfile:
                    version_logfile.write(version_log)
            except Exception:
                print("Logger could not write to file!")
                pass

        # start writing into the buffer
        self._session_buffer = "Starting at " \
            + self._start_time.strftime("%a %b %d %H:%M:%S %Z %Y UTC")+'\n'

        # set attributes
        self._logdir = logdir
        self._variables = []
        self._print_variables = set()

    def register_variables(self, variables, print_variables=None,
                           print_all=False):
        """
        Registers the variables which are supposed to get logged. This method
        can only be called once per session for correct behavior.

        Parameters
        ----------
        variables : list of strings
            A list which contains the names of all the variables which are
            supposed to get logged. Variables must be scalar.
        print_variables : list of strings
            The variables which are supposed to get printed on every step if
            Logger.log is called with printig=True. Must be a subset of
            variables.
        print_all : boolean
            If true make print_variables = variables. Overrides the
            print_variables parameter.
        """
        if print_all:
            print_variables = variables
        else:
            # verify that print_variables is a subset of variables
            if not set(print_variables) <= set(variables):
                raise NotASubsetError("print_variables must be a subset of "
                                      "variables.")
        self._variables = variables
        self._print_variables = set(print_variables)

        # init pandas dataframe buffer
        self._data_buffer = pd.DataFrame(columns=variables)

    # ----------------------- Writing to Files --------------------------------
    def write(self):
        """
        Writes the current buffers to files.

        Should only be used sparingly to not waist too many resources.
        """
        try:
            # write print buffer to session log
            with open(self._logdir+'session.log', 'a') as session_logfile:
                session_logfile.write(self._session_buffer)
            self._session_buffer = ""

            # write all variables to hdf5 store
            store = pd.HDFStore(self._logdir+'variables.hdf5')

            store.append('Variables', self._data_buffer, format='t',
                         data_columns=True)
            store.close()
        except Exception:
            print("Logger could not write to file!")
            pass

        # reset the data buffer TODO: better way?
        self._data_buffer = pd.DataFrame(columns=self._variables)

    # -------------------- Public Logging Methods -----------------------------
    def log(self, step, variables, printing=True):
        """
        Very basic and specialized logging method, no time left for today but
        this will be greatly improved.

        Parameters
        ----------
        step : integer
            The current training step.
        variables : list of scalar numbers
            A list with one value for each registered variable in the same
            order they have been registered. The variables have to be
            registered first using Logger.register_variables.
        printing : boolean
            Whether or not to print the logged step.
        """
        if len(variables) != len(self._variables):
            raise InvalidNumberOfVariables("The number of parsed variables "
                                           "does not equal the number of "
                                           "registered variables")

        session_time = datetime.utcnow() - self._start_time
        hours, remainder = divmod(session_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        line = ("[{:02d}:{:02d}:{:02d} {:08d}]").format(
            hours, minutes, seconds, step)
        for i, value in enumerate(variables):
            if self._variables[i] in self._print_variables:
                line += " {}: {:2.3f}".format(self._variables[i], value)
        line += '\n'

        # append row to data buffer
        self._data_buffer.loc[step] = variables

        if printing:
            print(line[:-1])

        self._session_buffer += line

    def print(self):
        """
        Prints the current session buffer contents without the last newline.
        """
        print(self._session_buffer[:-1])


# -------------------------------- Exceptions ---------------------------------
class LogdirAlreadyExistsError(Exception):
    pass


class NotASubsetError(Exception):
    pass


class InvalidNumberOfVariables(Exception):
    pass
