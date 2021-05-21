from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from optimal_model_selection import get_matches_data, transform_data
from utils import get_logger
from walk_operations import get_current_probability

logger = get_logger()


def plot_results(score_evolution):
    x_axis = range(1, len(score_evolution) + 1)
    plt.plot(x_axis, all_bets.naive_balance, 'b--', label='naive', linewidth=0.9)
    plt.plot(x_axis, all_bets.prob_balance, 'r-', label='probability', linewidth=0.9)
    plt.plot(x_axis, all_bets.odds_balance, 'y-.', label='1/odds', linewidth=0.9)
    plt.axis([0, len(all_bets), -4, 13])
    plt.xlabel('bet number')
    plt.ylabel('account balance')


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
    final_score = test_model(walks, starting_probabilities, all_matches_set_odds, c_lambda, model_type)
    logger.info(f"Final score after betting is {final_score}")


if __name__ == '__main__':
    main()
