from typing import List


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
