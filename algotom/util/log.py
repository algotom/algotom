'''
    tomoscan custom logger
    
'''
import logging

logger = logging.getLogger(__name__)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def setup_custom_logger(lfname=None, stream_to_console=True):

    logger.setLevel(logging.DEBUG)

    if (lfname != None):
        fHandler = logging.FileHandler(lfname)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        fHandler.setFormatter(file_formatter)
        logger.addHandler(fHandler)
    if stream_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredLogFormatter('%(asctime)s - %(message)s'))
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)

class ColoredLogFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None, style='%'):
        # Logging defines
        self.__GREEN = "\033[92m"
        self.__RED = '\033[91m'
        self.__YELLOW = '\033[33m'
        self.__ENDC = '\033[0m'
        super().__init__(fmt, datefmt, style)
    
    
    def formatMessage(self,record):
        if record.levelname=='INFO':
            record.message = self.__GREEN + record.message + self.__ENDC
        elif record.levelname == 'WARNING':
            record.message = self.__YELLOW + record.message + self.__ENDC
        elif record.levelname == 'ERROR':
            record.message = self.__RED + record.message + self.__ENDC
        return super().formatMessage(record)