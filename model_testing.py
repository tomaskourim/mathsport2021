from typing import List

import numpy as np

from optimal_model_selection import get_matches_data, transform_data
from utils import get_logger
from walk_operations import get_current_probability

logger = get_logger()


def evaluate_set(current_probability: float, odds: np.array, result: int) -> float:
    if current_probability > 1.1 / odds[0]:
        # virtual bet
        # always bet 1 unit
        return odds[0] - 1 if result == 1 else -1

        # bet current_probability units
        # return current_probability * odds[0] - current_probability if result == 1 else -current_probability

        # bet 1/odds units
        # return 1 - 1 / odds[0] if result == 1 else -1 / odds[0]
    else:
        # no bet
        return 0


def test_model(walks: List[List[int]], starting_probabilities: List[float], all_matches_set_odds: List[List[float]],
               c_lambda: float, model_type: str) -> float:
    final_score = 0
    for walk, starting_probability, single_match_set_odds in zip(walks, starting_probabilities, all_matches_set_odds):
        current_probability = starting_probability
        for i in range(1, len(walk)):
            current_probability = get_current_probability([c_lambda], current_probability, walk[i - 1], model_type)
            if current_probability >= 1 or current_probability <= 0:
                raise Exception(f"Probability out of bounds: {current_probability}")  # should not happen
            final_score = final_score + evaluate_set(current_probability, single_match_set_odds[i], walk[i])
    return final_score


def main():
    # get all matches from testing dataset
    start_date = '2021-05-01 00:00:00.000000'
    end_date = '2021-05-20 00:00:00.000000'
    matches_data = get_matches_data(start_date, end_date)

    # transform data
    walks, starting_probabilities, all_matches_set_odds = transform_data(matches_data)

    # compute probability and odds for each set
    c_lambdas, model_type = get_optimal_model()

    # compare with real odds
    final_score = test_model(walks, starting_probabilities, all_matches_set_odds, c_lambda, model_type)
    logger.info(f"Final score after betting is {final_score}")


if __name__ == '__main__':
    main()
