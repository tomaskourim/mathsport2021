import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt

from database_operations import execute_sql_postgres
from optimal_fair_odds_parameter import get_fair_odds_parameter, get_fair_odds
from utils import COLUMN_NAMES, ERROR_VALUE, OPTIMIZATION_ALGORITHM
from walk_operations import get_current_probability


def get_matches_data(start_date: str, end_date: str) -> pd.DataFrame:
    query = "SELECT matchid, home,     away,     set_number,     odd1,     odd2,     " \
            "CASE WHEN result = 'home' THEN 1 ELSE -1 END AS result,     start_time_utc FROM (     " \
            "SELECT *,         CASE             WHEN match_part = 'set1'                 THEN 1             " \
            "WHEN match_part = 'set2'                 THEN 2             " \
            "WHEN match_part = 'set3'                 THEN 3             " \
            "WHEN match_part = 'set4'                 THEN 4             " \
            "WHEN match_part = 'set5'                 THEN 5             " \
            "END AS set_number_odds     FROM odds) AS odds_enhanced          " \
            "INNER JOIN " \
            "(SELECT *, ma.id AS matchid  FROM matches_bookmaker mb           " \
            "JOIN matches ma ON mb.match_id = ma.id           " \
            "JOIN match_course mc ON mb.match_id = mc.match_id) AS match_course_enhanced " \
            "ON odds_enhanced.match_bookmaker_id = match_course_enhanced.match_bookmaker_id " \
            "AND     odds_enhanced.bookmaker_id = match_course_enhanced.bookmaker_id " \
            "AND     odds_enhanced.set_number_odds = match_course_enhanced.set_number" \
            " WHERE start_time_utc >= %s AND     " \
            "start_time_utc < %s " \
            "ORDER BY start_time_utc, match_course_enhanced.matchid, match_course_enhanced.set_number"
    return pd.DataFrame(execute_sql_postgres(query, [start_date, end_date], False, True), columns=COLUMN_NAMES)


def transform_data(matches_data: pd.DataFrame) -> Tuple[List[List[int]], List[float], List[np.array]]:
    # transform walks into List[List[int]
    # get p0 for each walk

    fair_odds_parameter = get_fair_odds_parameter()

    walks = []
    starting_probabilities = []
    matchid = ""
    walk = []
    all_matches_set_odds = []
    single_match_set_odds = []
    for index, set_data in matches_data.iterrows():
        if matchid != set_data.matchid:  # TODO maybe there are some matches with odds for set1 and set3, but not set2
            matchid = set_data.matchid
            first_set_odds = np.array([set_data.odd1, set_data.odd2])
            first_set_probabilities = get_fair_odds(first_set_odds, fair_odds_parameter)
            starting_probabilities.append(first_set_probabilities[0])
            walks.append(walk)
            walk = [set_data.result]
            all_matches_set_odds.append(single_match_set_odds)
            single_match_set_odds = [first_set_odds]
        else:
            walk.append(set_data.result)
            single_match_set_odds.append(np.array([set_data.odd1, set_data.odd2]))
    walks.append(walk)
    all_matches_set_odds.append(single_match_set_odds)
    walks = walks[1:]
    all_matches_set_odds = all_matches_set_odds[1:]

    index = 0
    while index < len(walks):
        if len(walks[index]) < 0:
            raise Exception("Walk length should not be less than 1")
        elif len(walks[index]) == 1:
            del walks[index]
            del starting_probabilities[index]
            del all_matches_set_odds[index]
        else:
            index = index + 1

    if len(walks) != len(starting_probabilities) or len(walks) != len(all_matches_set_odds):
        raise Exception("There has to be the same number of walks as starting probabilities and set odds.")

    return walks, starting_probabilities, all_matches_set_odds


def get_single_walk_log_likelihood(log_likelihood: float, c_lambdas: List[float], starting_probability: float,
                                   walk: List[int], model_type: str, starting_index: int) -> float:
    current_probability = starting_probability
    for i in range(starting_index, len(walk)):
        current_probability = get_current_probability(c_lambdas, current_probability, walk[i - 1], model_type)
        if current_probability >= 1 or current_probability <= 0 or max(c_lambdas) >= 1 or min(c_lambdas) <= 0:
            return -1e20  # so that I dont get double overflow error
        log_likelihood = log_likelihood + np.log(
            0.5 * ((1 + walk[i]) * current_probability + (1 - walk[i]) * (1 - current_probability)))
    return log_likelihood


def negative_log_likelihood_single_lambda(c_lambda: float, walks: List[List[int]], starting_probabilities: List[float],
                                          model_type: str) -> float:
    log_likelihood = 0
    for starting_probability, walk in zip(starting_probabilities, walks):
        log_likelihood = get_single_walk_log_likelihood(log_likelihood, [c_lambda],
                                                        starting_probability, walk,
                                                        model_type, starting_index=1)
    return -log_likelihood


def negative_log_likelihood_two_lambdas(c_lambdas: List[float], walks: List[List[int]],
                                        starting_probabilities: List[float],
                                        model_type: str) -> float:
    log_likelihood = 0
    for starting_probability, walk in zip(starting_probabilities, walks):
        log_likelihood = get_single_walk_log_likelihood(log_likelihood, c_lambdas,
                                                        starting_probability, walk,
                                                        model_type, starting_index=1)
    return -log_likelihood


def find_akaike_single(model: str, walks: List[List[int]], starting_probabilities: List[float], result: List[float],
                       current_model: str,
                       min_akaike: float) -> Tuple[float, List[float], str]:
    opt_result = opt.minimize_scalar(negative_log_likelihood_single_lambda, bounds=(0, 1), method='bounded',
                                     args=(walks, starting_probabilities, model))
    akaike = 2 + 2 * opt_result.fun
    if not opt_result.success:
        logging.error(f"Could not fit model {model}")
    if opt_result.success and akaike < min_akaike:
        min_akaike = akaike
        current_model = model
        result = opt_result.x
    return min_akaike, result, current_model


def find_akaike(guess: np.ndarray, model: str, walks: List[List[int]], starting_probabilities: List[float],
                result: List[float], current_model: str, min_akaike: float) -> Tuple[float, List[float], str]:
    opt_result = opt.minimize(negative_log_likelihood_two_lambdas, guess, method=OPTIMIZATION_ALGORITHM,
                              args=(walks, starting_probabilities, model))
    akaike = 2 * len(guess) + 2 * opt_result.fun
    if not opt_result.success:
        logging.error(f"Could not fit model {model}")
    if opt_result.success and akaike < min_akaike:
        min_akaike = akaike
        current_model = model
        result = opt_result.x
    return min_akaike, result, current_model


def get_model_estimate(walks: List[List[int]], starting_probabilities: List[float]) -> Tuple[List[float], str]:
    """
    Uses the Akaike information criterion
    https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size
    to get the optimal model.
    AIC = 2k - 2ln(L)
    opt.minimize() returns directly -ln(L)
    :param walks:
    :param starting_probabilities:
    :return:
    """
    result = [ERROR_VALUE, ERROR_VALUE]
    current_model = ERROR_VALUE
    min_akaike = 1e20  # big enough

    # single lambda models
    model = 'success_punished'
    min_akaike, result, current_model = find_akaike_single(model, walks, starting_probabilities, result, current_model,
                                                           min_akaike)

    model = 'success_rewarded'
    min_akaike, result, current_model = find_akaike_single(model, walks, starting_probabilities, result, current_model,
                                                           min_akaike)

    # two lambdas models
    guess = np.repeat(0.9, 2)
    model = 'success_punished_two_lambdas'
    min_akaike, result, current_model = find_akaike(guess, model, walks, starting_probabilities, result, current_model,
                                                    min_akaike)

    model = 'success_rewarded_two_lambdas'
    min_akaike, result, current_model = find_akaike(guess, model, walks, starting_probabilities, result, current_model,
                                                    min_akaike)

    return result, current_model


def get_optimal_model() -> Tuple[List[float], str]:
    # get all data
    start_date = '2021-02-01 00:00:00.000000'
    end_date = '2021-05-01 00:00:00.000000'
    matches_data = get_matches_data(start_date, end_date)

    # transform data
    walks, starting_probabilities, _ = transform_data(matches_data)
    logging.info(f"There are {len(walks)} walks available.")

    # get model estimate + parameters
    return get_model_estimate(walks, starting_probabilities)


def main():
    c_lambdas, model_type = get_optimal_model()
    logging.info(f"Optimal mode type is {model_type} with lambda = {c_lambdas}.")

    # repeat for subgroups


if __name__ == '__main__':
    main()
