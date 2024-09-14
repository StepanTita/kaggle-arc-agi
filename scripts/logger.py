import logging
import os
import traceback

from utils import grey, cyan, yellow, red, bold


class CustomTerminalFormatter(logging.Formatter):
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    reset = '\033[0m'

    FORMATS = {
        logging.DEBUG: grey(format) + reset,
        logging.INFO: cyan(format) + reset,
        logging.WARNING: yellow(format) + reset,
        logging.ERROR: red(format) + reset,
        logging.CRITICAL: bold(red(format)) + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomFileFormatter(logging.Formatter):
    def format(self, record):
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        return formatter.format(record)


class SelectiveLogger:
    def __init__(self, terminal_logger, file_logger):
        self.term = terminal_logger
        self.file = file_logger

    def info(self, msg='', terminal=True):
        if terminal:
            self.term.info(msg)
        self.file.info(msg)

    def debug(self, msg='', terminal=True):
        if terminal:
            self.term.debug(msg)
        self.file.debug(msg)

    def warn(self, msg='', terminal=True):
        if terminal:
            self.term.warn(msg)
        self.file.warn(msg)

    def error(self, msg='', terminal=True):
        if terminal:
            self.term.error(msg)
        self.file.error(msg)

    def critical(self, msg='', terminal=True):
        if terminal:
            self.term.critical(msg)
        self.file.critical(msg)


class SingletonStreamHandler(logging.StreamHandler):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonStreamHandler, cls).__new__(cls)
            cls._instance.__init_once()
        return cls._instance

    def __init_once(self):
        super(SingletonStreamHandler, self).__init__()


class SingletonFileHandler(logging.FileHandler):
    _instances = {}

    def __new__(cls, filename):
        if filename not in cls._instances:
            cls._instances[filename] = super(SingletonFileHandler, cls).__new__(cls)
            cls._instances[filename].__init_once(filename)
        return cls._instances[filename]

    def __init_once(self, filename):
        super(SingletonFileHandler, self).__init__(filename)


def get_logger(log_path, log_file, log_name='default'):
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    term = logging.getLogger(log_name).getChild('terminal')
    file = logging.getLogger(log_name).getChild('file')

    # Get singleton handlers
    sh = SingletonStreamHandler()
    fh = SingletonFileHandler(f'{log_path}/{log_file.replace("/", "_")}.log')

    fh.setFormatter(CustomFileFormatter())

    term.handlers = []
    file.handlers = []

    term.addHandler(sh)
    file.addHandler(fh)

    # Set logging level to the logger
    term.setLevel(logging.DEBUG)
    file.setLevel(logging.DEBUG)

    return SelectiveLogger(term, file)


def log_continue(fn):
    def wrapped(log, *args, **kwargs):
        try:
            return fn(log, *args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
            log.error(str(e))
            return None

    return wrapped
