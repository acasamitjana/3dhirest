from scipy.spatial.distance import dice

def dice_coefficient(y_true, y_pred):
    """
    Computes the Sorensen-Dice metric
                    TP
        Dice = 2 -------
                  T + P
    Parameters
    ----------
    y_true : numpy.array
        Binary representation
    y_pred : keras.placeholder
        Binary representation
    Returns
    -------
    scalar
        Dice metric
    """

    y_pred = y_pred > 0
    y_true = y_true > 0

    y_pred_flatten = y_pred.reshape(-1,1)
    y_true_flatten = y_true.reshape(-1,1)

    dice_score_negated = dice(y_true_flatten, y_pred_flatten)

    return 1 - dice_score_negated

