import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt

from database_operations import execute_sql_postgres

EVEN_ODDS_PROBABILITY = 0.5
COLUMN_NAMES = ['home', 'away', 'set_number', 'odd1', 'odd2', 'result', 'start_time_utc']


def get_fair_odds(odds: np.ndarray, fair_odds_parameter: float) -> np.ndarray:
    odds_probability_norm = sum(1 / odds)
    normalized_odds_probabilities = 1 / (odds * odds_probability_norm)
    odds_weights = (1 - normalized_odds_probabilities) + (
            normalized_odds_probabilities - EVEN_ODDS_PROBABILITY) * fair_odds_parameter
    probabilities = 1 / odds - odds_weights * (odds_probability_norm - 1)

    return probabilities


# just for the sake of minimization
def negative_log_likelihood_fair_odds_parameter(fair_odds_parameter: float, matches_data: pd.DataFrame) -> float:
    return - log_likelihood_fair_odds_parameter(fair_odds_parameter, matches_data)


def log_likelihood_fair_odds_parameter(fair_odds_parameter: float, matches_data: pd.DataFrame) -> float:
    log_likelihood = 0
    for index, match_data in matches_data.iterrows():
        odds = np.array([match_data["odd1"], match_data["odd2"]])
        probabilities = get_fair_odds(odds, fair_odds_parameter)
        if probabilities[0] + probabilities[1] < 0.999999999999999 or \
                probabilities[0] + probabilities[1] > 1.00000000000001:
            raise Exception("Probabilities do not sum to 1")
        result = match_data["result"]
        log_likelihood = log_likelihood + np.log(result * probabilities[0] + (1 - result) * (1 - probabilities[0]))

    return log_likelihood


def find_fair_odds_parameter(training_set: pd.DataFrame) -> Optional[float]:
    opt_result = opt.minimize_scalar(negative_log_likelihood_fair_odds_parameter, bounds=(0, 2), method='bounded',
                                     args=training_set)
    if opt_result.success:
        logging.debug("Fitted successfully.")
        return opt_result.x
    else:
        return None


def get_first_set_data(start_date: str, end_date: str) -> pd.DataFrame:
    query = "SELECT home,     away,     set_number,     odd1,     odd2,     case when result = 'home' then 1 else 0 end as result,     start_time_utc FROM (     SELECT *,         CASE             WHEN match_part = 'set1'                 THEN 1             WHEN match_part = 'set2'                 THEN 2             WHEN match_part = 'set3'                 THEN 3             WHEN match_part = 'set4'                 THEN 4             WHEN match_part = 'set5'                 THEN 5             END AS set_number_odds     FROM odds) AS odds_enhanced          INNER JOIN (SELECT *, ma.id AS matchid  FROM matches_bookmaker mb           JOIN matches ma ON mb.match_id = ma.id           JOIN match_course mc ON mb.match_id = mc.match_id join tournament t on ma.tournament_id = t.id) AS match_course_enhanced ON odds_enhanced.match_bookmaker_id = match_course_enhanced.match_bookmaker_id AND     odds_enhanced.bookmaker_id = match_course_enhanced.bookmaker_id AND     odds_enhanced.set_number_odds = match_course_enhanced.set_number " \
            "WHERE start_time_utc >= %s AND     " \
            "start_time_utc < %s AND set_number = 1 and sex='women' and type='singles' ORDER BY match_course_enhanced.matchid, match_course_enhanced.set_number"
    column_names = ["home", "away", "set_number", "odd1", "odd2", "result", "start_time_utc"]
    return pd.DataFrame(execute_sql_postgres(query, [start_date, end_date], False, True), columns=column_names)


def transform_home_favorite(matches_data: pd.DataFrame) -> pd.DataFrame:
    transformed_matches = []
    for _, match_data in matches_data.iterrows():
        if match_data.odd1 <= match_data.odd2:
            transformed_matches.append(list(match_data))
        else:
            transformed_matches.append(list(transform_home_favorite_single(match_data)))

    transformed_matches = pd.DataFrame(transformed_matches, columns=COLUMN_NAMES)

    return transformed_matches


def transform_home_favorite_single(match_data: pd.Series) -> pd.Series:
    transformed_data = pd.Series(index=COLUMN_NAMES)
    transformed_data.home = match_data.away
    transformed_data.away = match_data.home
    transformed_data.set_number = match_data.set_number
    transformed_data.odd1 = match_data.odd2
    transformed_data.odd2 = match_data.odd1
    transformed_data.result = 0 if match_data.result else 1
    transformed_data.start_time_utc = match_data.start_time_utc

    return transformed_data


def main():
    start_date = '2021-02-01 00:00:00.000000'
    end_date = '2021-05-01 00:00:00.000000'
    training_set = get_first_set_data(start_date, end_date)
    fair_odds_parameter = find_fair_odds_parameter(training_set)
    logger.info(f"Optimal fair odds parameter is {fair_odds_parameter}")

    # training_set = transform_home_favorite(training_set)
    # fair_odds_parameter = find_fair_odds_parameter(training_set)
    # logger.info(f"Optimal fair odds parameter is {fair_odds_parameter}")


if __name__ == '__main__':
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

    main()
