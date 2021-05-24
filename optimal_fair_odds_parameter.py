import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt

from database_operations import execute_sql_postgres
from utils import COLUMN_NAMES

EVEN_ODDS_PROBABILITY = 0.5


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
    query = "SELECT matchid, home,     away,     set_number,     odd1,     odd2,     " \
            "CASE WHEN result = 'home' THEN 1 ELSE 0 END AS result,     start_time_utc FROM (     " \
            "SELECT *,         CASE             " \
            "WHEN match_part = 'set1'                 THEN 1             " \
            "WHEN match_part = 'set2'                 THEN 2             " \
            "WHEN match_part = 'set3'                 THEN 3             " \
            "WHEN match_part = 'set4'                 THEN 4             " \
            "WHEN match_part = 'set5'                 THEN 5             " \
            "END AS set_number_odds     FROM odds) AS odds_enhanced          " \
            "INNER JOIN (SELECT *, ma.id AS matchid  FROM matches_bookmaker mb           " \
            "JOIN matches ma ON mb.match_id = ma.id           " \
            "JOIN match_course mc ON mb.match_id = mc.match_id JOIN tournament t ON ma.tournament_id = t.id) " \
            "AS match_course_enhanced ON odds_enhanced.match_bookmaker_id = match_course_enhanced.match_bookmaker_id " \
            "AND     odds_enhanced.bookmaker_id = match_course_enhanced.bookmaker_id " \
            "AND     odds_enhanced.set_number_odds = match_course_enhanced.set_number " \
            "WHERE start_time_utc >= %s AND     " \
            "start_time_utc < %s AND set_number = 1 " \
            "ORDER BY start_time_utc, match_course_enhanced.matchid, match_course_enhanced.set_number"
    # and sex='women' and type='singles'

    return pd.DataFrame(execute_sql_postgres(query, [start_date, end_date], False, True), columns=COLUMN_NAMES)


def get_fair_odds_parameter() -> float:
    start_date = '2021-02-01 00:00:00.000000'
    end_date = '2021-05-01 00:00:00.000000'
    training_set = get_first_set_data(start_date, end_date)
    return find_fair_odds_parameter(training_set)


def main():
    fair_odds_parameter = get_fair_odds_parameter()
    logging.info(f"Optimal fair odds parameter is {fair_odds_parameter}")


if __name__ == '__main__':
    main()
