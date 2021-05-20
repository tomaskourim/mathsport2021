from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt

from database_operations import execute_sql_postgres
from odds_to_probabilities import get_fair_odds_parameter, get_fair_odds
from utils import get_logger, COLUMN_NAMES, ERROR_VALUE, OPTIMIZATION_ALGORITHM


def get_matches_data(start_date: str, end_date: str) -> pd.DataFrame:
    query = "SELECT matchid, home,     away,     set_number,     odd1,     odd2,     " \
            "case when result = 'home' then 1 else 0 end as result,     start_time_utc FROM (     " \
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


def transform_data(matches_data: pd.DataFrame) -> Tuple[List[List[int]], List[float]]:
    # transform walks into List[List[int]
    # get p0 for each walk

    fair_odds_parameter = get_fair_odds_parameter()

    walks = []
    starting_probabilities = []
    matchid = ""
    walk = []
    for index, set_data in matches_data.iterrows():
        if matchid != set_data.matchid:
            matchid = set_data.matchid
            odds = np.array([set_data.odd1, set_data.odd2])
            probabilities = get_fair_odds(odds, fair_odds_parameter)
            starting_probabilities.append(probabilities[0])
            walks.append(walk)
            walk = [set_data.result]
        else:
            walk.append(set_data.result)
    walks.append(walk)
    walks = walks[1:]

    index = 0
    while index < len(walks):
        if len(walks[index]) < 0:
            raise Exception("Walk length should not be less than 1")
        elif len(walks[index]) == 1:
            del walks[index]
            del starting_probabilities[index]
        else:
            index = index + 1

    if len(walks) != len(starting_probabilities):
        raise Exception("There has to be the same number of walks as starting probabilities.")

    return walks, starting_probabilities


def get_current_probability(c_lambdas: List[float], last_probability: float, step: int, walk_type: str) -> float:
    """
    Computes the transition probability for the next step according to the respective definition as in the paper.
    :param c_lambdas:
    :param last_probability:
    :param step: as Ising variable
    :param walk_type:
    :return:
    """
    if step == '' or step == 0:  # at the beginning of the walk just return p0
        return last_probability
    if walk_type == 'success_punished':
        return c_lambdas[0] * last_probability + (0.5 * (1 - c_lambdas[0]) * (1 - step))
    elif walk_type == 'success_rewarded':
        return (c_lambdas[0]) * last_probability + (0.5 * (1 - c_lambdas[0]) * (1 + step))
    elif walk_type == 'success_punished_two_lambdas':
        return 0.5 * (((1 + step) * c_lambdas[0]) * last_probability + (1 - step) * (
                1 - (c_lambdas[1]) * (1 - last_probability)))
    elif walk_type == 'success_rewarded_two_lambdas':
        return 0.5 * (((1 - step) * c_lambdas[0]) * last_probability + ((1 + step) * (
                1 - (c_lambdas[1]) * (1 - last_probability))))
    else:
        raise Exception(f'Unexpected walk type: {walk_type}')


def get_single_walk_log_likelihood(log_likelihood: float, c_lambdas: List[float], starting_probability: float,
                                   walk: List[int], model_type: str, starting_index: int) -> float:
    current_probability = starting_probability
    for i in range(starting_index, len(walk)):
        current_probability = get_current_probability(c_lambdas, current_probability, walk[i - 1], model_type)
        if current_probability >= 1 or current_probability <= 0 or max(c_lambdas) >= 1 or min(c_lambdas) <= 0:
            return -1e20  # so that I dont get double overflow error
        log_likelihood = log_likelihood + np.log(
            walk[i] * current_probability + (1 - walk[i]) * (1 - current_probability))
    return log_likelihood


def negative_log_likelihood_single_lambda(c_lambda: float, walks: List[List[int]], starting_probabilities: List[float],
                                          model_type: str) -> float:
    log_likelihood = 0
    for starting_probability, walk in zip(starting_probabilities, walks):
        log_likelihood = get_single_walk_log_likelihood(log_likelihood, [c_lambda], starting_probability, walk,
                                                        model_type, starting_index=1)
    return -log_likelihood


def negative_log_likelihood_two_lambdas(c_lambdas: List[float], walks: List[List[int]],
                                        starting_probabilities: List[float],
                                        model_type: str) -> float:
    log_likelihood = 0
    for starting_probability, walk in zip(starting_probabilities, walks):
        log_likelihood = get_single_walk_log_likelihood(log_likelihood, c_lambdas, starting_probability, walk,
                                                        model_type, starting_index=1)
    return -log_likelihood


def find_akaike_single(model: str, walks: List[List[int]], starting_probabilities: List[float], result: List[float],
                       current_model: str,
                       min_akaike: float) -> Tuple[float, List[float], str]:
    opt_result = opt.minimize_scalar(negative_log_likelihood_single_lambda, bounds=(0, 1), method='bounded',
                                     args=(walks, starting_probabilities, model))
    akaike = 2 + 2 * opt_result.fun
    if not opt_result.success:
        logger.error(f"Could not fit model {model}")
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
        logger.error(f"Could not fit model {model}")
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
    guess = np.repeat(0.5, 2)
    model = 'success_punished_two_lambdas'
    min_akaike, result, current_model = find_akaike(guess, model, walks, starting_probabilities, result, current_model,
                                                    min_akaike)

    model = 'success_rewarded_two_lambdas'
    min_akaike, result, current_model = find_akaike(guess, model, walks, starting_probabilities, result, current_model,
                                                    min_akaike)

    return result, current_model


def main():
    # get all data
    start_date = '2021-02-01 00:00:00.000000'
    end_date = '2021-05-01 00:00:00.000000'
    matches_data = get_matches_data(start_date, end_date)

    # transform data
    walks, starting_probabilities = transform_data(matches_data)
    logger.info(f"There are {len(walks)} walks available.")

    # get model estimate + parameters
    c_lambdas, model_type = get_model_estimate(walks, starting_probabilities)

    # repeat for subgroups
    logger.info("Done")


if __name__ == '__main__':
    logger = get_logger()
    main()
