import logging

COLUMN_NAMES = ["matchid", "home", "away", "set_number", "odd1", "odd2", "result", "start_time_utc"]
ERROR_VALUE = 'not_fitted'
OPTIMIZATION_ALGORITHM = 'Nelder-Mead'


def get_logger() -> logging.Logger:
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel('DEBUG')

    # Create handlers
    total_handler = logging.FileHandler('logfile_total.log', mode='w', encoding="utf-8")
    info_handler = logging.FileHandler('logfile_info.log', encoding="utf-8")
    error_handler = logging.FileHandler('logfile_error.log', encoding="utf-8")
    stdout_handler = logging.StreamHandler()

    total_handler.setLevel(logging.DEBUG)
    info_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.WARNING)
    stdout_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    logging_format = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(message)s')
    total_handler.setFormatter(logging_format)
    info_handler.setFormatter(logging_format)
    error_handler.setFormatter(logging_format)
    stdout_handler.setFormatter(logging_format)

    # Add handlers to the logger
    logger.addHandler(total_handler)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(stdout_handler)

    return logger
