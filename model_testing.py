import logging
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from optimal_model_selection import get_matches_data, transform_data
from walk_operations import get_current_probability

import scipy.stats as stat


def get_global_extremes_coordinates(array: np.ndarray) -> Tuple[float, float, Tuple[int, float]]:
    minimum = min(array)
    min_coordinates = (np.where(array == minimum)[0][0], minimum)
    return minimum, max(array), min_coordinates


def get_expected_results(expected_wins: np.ndarray, variance_wins: np.ndarray) -> Tuple[float, float]:
    return sum(expected_wins), math.sqrt(sum(variance_wins))


def get_p_value(computing_type: str, observed_values: np.ndarray, expected_values: np.ndarray,
                variances: np.ndarray) -> float:
    logging.info(f"----------------------------------------\n\t\tTesting {computing_type}:")
    number_observations = len(observed_values)
    x_mean = sum(observed_values) / number_observations
    mu_hat = sum(expected_values) / number_observations
    var_hat = sum(variances) / number_observations
    logging.info(
        f"Observations: {number_observations}. \
        Observed value: {x_mean:.3f}, expected value: {mu_hat:.3f}, standard deviation: {math.sqrt(var_hat):.3f}")
    expected_distribution = stat.norm()

    observed_value = math.sqrt(number_observations) * (x_mean - mu_hat) / math.sqrt(var_hat)

    cdf_observed = expected_distribution.cdf(observed_value)
    p_value = min(cdf_observed, 1 - cdf_observed) * 2

    logging.info(f"P-value: {p_value:.3f}")

    if p_value < 0.1:
        logging.info("Reject H0 on 90% level.")
    else:
        logging.info("Cannot reject H0.")

    if p_value < 0.05:
        logging.info("Reject H0 on 95% level.")

    if p_value < 0.01:
        logging.info("Reject H0 on 99% level.")

    return p_value


def log_result(betting_type: str, minimum: float, maximum: float, final_balance: float, expected_win: float,
               standard_deviation: float):
    logging.info(
        f"{betting_type}: \
        Min = {minimum:.2f}; \
        Max = {maximum:.2f}; \
        Profit: {final_balance:.2f}; \
        ROI: {final_balance / abs(minimum):.2f} \
        E_win: {expected_win:.2f}; \
        Std_dev: {standard_deviation:.2f}.")


def plot_results(all_bets: pd.DataFrame):
    number_bets = len(all_bets)
    x_axis = range(1, len(all_bets) + 1)
    plt.plot(x_axis, all_bets.naive_balance, 'b--', label='naive', linewidth=0.9)
    plt.plot(x_axis, all_bets.prob_balance, 'r-', label='probability', linewidth=0.9)
    plt.plot(x_axis, all_bets.odds_balance, 'y-.', label='1/odds', linewidth=0.9)
    plt.axis([0, len(all_bets), -4, 13])
    plt.xlabel('bet number')
    plt.ylabel('account balance')

    naive_min, naive_max, naive_min_coordinates = get_global_extremes_coordinates(all_bets.naive_balance)
    naive_expected_win, naive_variance = get_expected_results(all_bets.naive_expected_wins,
                                                              all_bets.naive_variance_wins)
    naive_min_annotation_coordinates = (naive_min_coordinates[0] - 60, naive_min_coordinates[1] + 0.3)
    # plt.annotate('global min naive', xy=naive_min_coordinates, xytext=naive_min_annotation_coordinates,
    #              arrowprops=dict(facecolor='black', shrink=0.01, width=1),
    #              )

    prob_min, prob_max, prob_min_coordinates = get_global_extremes_coordinates(all_bets.prob_balance)
    prob_expected_win, prob_variance = get_expected_results(all_bets.prob_expected_wins,
                                                            all_bets.prob_variance_wins)
    prob_min_annotation_coordinates = (prob_min_coordinates[0] - 90, prob_min_coordinates[1] - 1)
    # plt.annotate('global min probability and 1/odds', xy=prob_min_coordinates, xytext=prob_min_annotation_coordinates,
    #              arrowprops=dict(facecolor='black', shrink=0.01, width=1),
    #              )

    odds_min, odds_max, _ = get_global_extremes_coordinates(all_bets.odds_balance)
    odds_expected_win, odds_std_dev = get_expected_results(all_bets.odds_expected_wins,
                                                           all_bets.odds_variance_wins)

    plt.axhline(linewidth=0.5, color='k')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(7, 4.5)
    fig.show()
    fig.savefig('account_balance_development.pdf', bbox_inches='tight', dpi=300)

    log_result("Naiv betting", naive_min, naive_max, all_bets.naive_balance[number_bets - 1], naive_expected_win,
               naive_variance)
    log_result("Prob betting", prob_min, prob_max, all_bets.prob_balance[number_bets - 1], prob_expected_win,
               prob_variance)
    log_result("Odds betting", odds_min, odds_max, all_bets.odds_balance[number_bets - 1], odds_expected_win,
               odds_std_dev)

    # get_p_value("result", all_bets.result, all_bets.probability, all_bets.probability * (1 - all_bets.probability))
    get_p_value("naive", all_bets.naive_wins, all_bets.naive_expected_wins, all_bets.naive_variance_wins)
    get_p_value("prob", all_bets.prob_wins, all_bets.prob_expected_wins, all_bets.prob_variance_wins)
    get_p_value("odds", all_bets.odds_wins, all_bets.odds_expected_wins, all_bets.odds_variance_wins)


def test_model(walks: List[List[int]], starting_probabilities: List[float], all_matches_set_odds: List[List[float]],
               c_lambda: float, model_type: str) -> pd.DataFrame:
    coefficient = 1

    betting_win_naive = []
    betting_win_prob = []
    betting_win_odds = []

    expected_betting_win_naive = []
    expected_betting_win_prob = []
    expected_betting_win_odds = []

    variance_betting_win_naive = []
    variance_betting_win_prob = []
    variance_betting_win_odds = []

    balance_naive = [0]
    balance_prob = [0]
    balance_odds = [0]

    index = 0

    for walk, starting_probability, single_match_set_odds in zip(walks, starting_probabilities, all_matches_set_odds):
        current_probability = starting_probability
        for i in range(1, len(walk)):
            current_probability = get_current_probability([c_lambda], current_probability, walk[i - 1], model_type)
            if current_probability >= 1 or current_probability <= 0:
                raise Exception(f"Probability out of bounds: {current_probability}")  # should not happen

            # virtual betting
            # bet home
            if current_probability > coefficient * 1 / single_match_set_odds[i][0]:
                betting_probability = current_probability
                betting_odds = single_match_set_odds[i][0]
                bet_won = 1 if walk[i] == 1 else 0
            # bet away
            elif (1 - current_probability) > coefficient * 1 / single_match_set_odds[i][1]:
                betting_probability = 1 - current_probability
                betting_odds = single_match_set_odds[i][1]
                bet_won = 1 if walk[i] == -1 else 0
            # do not bet
            else:
                continue

            # always bet 1 unit
            naive_win = betting_odds - 1 if bet_won else -1
            betting_win_naive.append(naive_win)

            # bet current_probability units
            prob_win = betting_probability * betting_odds - betting_probability if bet_won else -betting_probability
            betting_win_prob.append(prob_win)

            # bet 1/odds units
            odds_win = 1 - 1 / betting_odds if bet_won else -1 / betting_odds
            betting_win_odds.append(odds_win)

            balance_naive.append(balance_naive[index] + betting_win_naive[index])
            balance_prob.append(balance_prob[index] + betting_win_prob[index])
            balance_odds.append(balance_odds[index] + betting_win_odds[index])

            expected_betting_win_naive.append(betting_probability * betting_odds - 1)
            expected_betting_win_prob.append(betting_probability * (betting_probability * betting_odds - 1))
            expected_betting_win_odds.append(betting_probability - 1 / betting_odds)

            variance_betting_win_naive.append(betting_probability * (betting_odds ** 2) * (1 - betting_probability))
            variance_betting_win_prob.append(
                (betting_probability ** 3) * (betting_odds ** 2) * (1 - betting_probability))
            variance_betting_win_odds.append(betting_probability * (1 - betting_probability))

            index = index + 1
            # end walk
        # end walks

    all_bets = pd.DataFrame()

    all_bets.insert(0, "naive_balance", balance_naive[1:], True)
    all_bets.insert(0, "naive_expected_wins", expected_betting_win_naive, True)
    all_bets.insert(0, "naive_variance_wins", variance_betting_win_naive, True)
    all_bets.insert(0, "naive_wins", betting_win_naive, True)

    all_bets.insert(0, "prob_balance", balance_prob[1:], True)
    all_bets.insert(0, "prob_expected_wins", expected_betting_win_prob, True)
    all_bets.insert(0, "prob_variance_wins", variance_betting_win_prob, True)
    all_bets.insert(0, "prob_wins", betting_win_prob, True)

    all_bets.insert(0, "odds_balance", balance_odds[1:], True)
    all_bets.insert(0, "odds_expected_wins", expected_betting_win_odds, True)
    all_bets.insert(0, "odds_variance_wins", variance_betting_win_odds, True)
    all_bets.insert(0, "odds_wins", betting_win_odds, True)

    return all_bets


def main():
    # get all matches from testing dataset
    start_date = '2021-05-01 00:00:00.000000'
    end_date = '2021-05-20 00:00:00.000000'
    matches_data = get_matches_data(start_date, end_date)

    # transform data
    walks, starting_probabilities, all_matches_set_odds = transform_data(matches_data)

    # compute probability and odds for each set
    # c_lambda, model_type = get_optimal_model()
    c_lambda, model_type = 0.8262665695105103, 'success_rewarded'

    # compare with real odds
    all_bets = test_model(walks, starting_probabilities, all_matches_set_odds, c_lambda, model_type)
    plot_results(all_bets)
    # logger.info(f"Final score after betting is {final_score}")


if __name__ == '__main__':
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()
