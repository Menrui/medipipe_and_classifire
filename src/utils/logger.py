import logging


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)

    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.CRITICAL + 1)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL + 1)

    return logger
