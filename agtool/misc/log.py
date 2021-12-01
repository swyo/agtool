import logging

opt = {
    'NOTSET': logging.NOTSET, 'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
    'WARN': logging.WARN, 'ERROR': logging.ERROR, 'CRITICAL': logging.CRITICAL
}


def _logging_format():
    fmt = '[%(levelname)-8s] %(asctime)s [%(filename)s] [%(funcName)s:%(lineno)d] %(message)s'
    return fmt


def _logging_datefmt():
    return '%Y-%m-%d %H:%M:%S'


basic_fmt = _logging_format()
basic_datefmt = _logging_datefmt()


def disable(level='INFO'):
    """Disable larger level.
    Args:
        level: selected among ('NOTSET', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL')

    Example:
        >>> disable(INFO)
        >>> Activated logging levels: ('WARN', 'ERROR', 'CRITICAL')
    """
    selected = opt[level]
    logging.disable(selected)
    levels, integers = list(zip(*sorted(opt.items(), key=lambda x: x[1])))
    for i, _integer in enumerate(integers[:-1]):
        if selected == _integer:
            activated = levels[i + 1:]
            print(f"Activated logging levels: {activated}")
            return


def get_logger(
    logger_name, level='WARN', filename=None,
    fmt=basic_fmt, datefmt=basic_datefmt, propagate=False
):
    """Get logger for logging.

    Args:
        logger_name: the name of logger
        level: logging level (default: INFO)
            level can be among ('DEBUG', 'INFO', 'WARN')
        filename: select to use file type logger (default: stream logger)

    Kwargs:
        fmt: format for logging.
        datefmt: date format for logging.
        propagate: propagete rule for logging.
    """
    assert level in opt.keys()
    logger = logging.getLogger(logger_name)
    logger.propagate = propagate
    logger.setLevel(level=opt[level])
    if filename:
        handler = logging.FileHandler(filename)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger
