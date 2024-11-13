import numpy as np
# Note: do not import additional libraries to implement these functions

#is this all binary
# how do late days work
def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    You do not need to implement confusion matrices for labels with more
    classes. You can assume this will always be a 2x2 matrix.

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    else:
        #check when its non binary
        return np.array([[np.sum((actual == 0) & (predictions == 0)), np.sum((actual == 0) & (predictions == 1)) ],
         [np.sum((actual == 1) & (predictions == 0)), np.sum((actual == 1) & (predictions == 1))]])




def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    return (np.sum((actual == 1) & (predictions == 1)) + np.sum((actual == 0) & (predictions == 0)))/predictions.shape[0]


def compute_precision_and_recall(actual, predictions): 
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output a tuple containing:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    #walk through logic of this
    if ((np.sum((actual == 1) & (predictions == 1)) + np.sum((actual == 1) & (predictions == 0)) )== 0 ):
        rec = np.nan
    else:
        rec = np.sum((actual == 1) & (predictions == 1))/(np.sum((actual == 1) & (predictions == 1)) + np.sum((actual == 1) & (predictions == 0)) )


    if ((np.sum((actual == 1) & (predictions == 1)) + np.sum((actual == 0) & (predictions == 1)) )== 0 ):
        prec = np.nan
        

    else:
        prec = np.sum((actual == 1) & (predictions == 1))/(np.sum((actual == 1) & (predictions == 1)) + np.sum((actual == 0) & (predictions == 1)) )
    return(prec, rec)


def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")


    print("hello")
    prec, rec = compute_precision_and_recall(actual, predictions)
    if((prec == np.nan or rec == np.nan)):
        return np.nan
    else:
        return 2*(prec*rec)/(prec + rec)


