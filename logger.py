import datetime
import sys


class LOGGER(object):

    def __init__(self, log_path=None):
        self.log_path = log_path

    def log(self, message):
        # put output into frame
        datetime_string = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        str_to_print = "%s\t%s\n"
        to_print = str_to_print % (datetime_string, str(message))

        # print to screen
        sys.stdout.write(to_print)
        sys.stdout.flush()

        # write to log file
        if self.log_path is not None:
            with open(self.log_path, 'a') as f:
                f.write(to_print)
