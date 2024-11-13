import numpy as np
import pandas as pd


def load_data(data_path):
    """
    This function loads the data in data_path csv into two numpy arrays:
    features (of size NxK) and labels (of size Nx1) where N is the number of rows
    and K is the number of features.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        labels (np.array): numpy array of size 1xN containing the N labels.
        attribute_names (list): list of strings containing names of each attribute
            (headers of csv)
    """
    if data_path.endswith('gz'):
        df = pd.read_csv(data_path, compression='gzip')
    else:
        df = pd.read_csv(data_path)

    feature_columns = [col for col in df.columns if col != "class"]
    features = df[feature_columns].to_numpy()
    label = df[["class"]].to_numpy()

    return features, label, feature_columns


def train_test_split(features, labels, fraction):
    """
    Split features and labels into training and testing. The first M points
    from the data will be used for training and the remaining
    (features.shape[0] - M) points will be used for testing. Where M is:

        M = int(features.shape[0] * fraction)

    However, when fraction is 1.0, both training and test splits are
    the entire dataset. Code for this special case is provided for you.

    Args:
        features (np.array): NxD numpy array containing D features for each example
        labels (np.array): Nx1 numpy array containing labels corresponding to each example
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns (a tuple containing four variables):
        train_features: MxD numpy array of examples to be used for training
        train_labels: Mx1 numpy array of labels corresponding to `train_features`
        test_features: (N - M)xD numpy array of examples to be used for testing
        test_labels: (N - M)x1 numpy array of labels corresponding to `test_features`
    """
    M = int(features.shape[0] * fraction)

    if fraction == 1.0:
        return (features, labels, features, labels)
    elif fraction < 1.0:
        return (features[0:M,:], labels[0:M], features[M:,:], labels[M:])
    else:
        raise ValueError('fraction must be less than or equal to 1.0!')


def cross_validation(features, labels, n_folds):
    """
    Split the data in `n_folds` different groups for cross-validation.
        Split the features and labels into a `n_folds` number of groups that
        divide the data as evenly as possible. Then for each group,
        return a tuple that treats that group as the test set and all
        other groups combine to make the training set.

        Note that this should be *deterministic*; don't shuffle the data.
        If there are 100 examples and you have 5 folds, each group
        should contain 20 examples and the first group should contain
        the first 20 examples.

        See test_cross_validation for expected behavior.

    Args:
        features: an NxK matrix of N examples, each with K features
        labels: an Nx1 array of N labels
        n_folds: the number of cross-validation groups

    Output:
        A list of tuples, where each tuple contains:
          (train_features, train_labels, test_features, test_labels)
    """

    assert features.shape[0] == labels.shape[0]

    per = int(features.shape[0]/n_folds)
    rem = features.shape[0] % n_folds
    arr = [per]*n_folds
    if (rem > 0) & (rem/ per < 0.75):
        for i in (range(0,features.shape[0],int(features.shape[0]/rem))):
            arr[i] = arr[i] + 1

    lst = []
    for i in range(n_folds):


        test_features = features[sum(arr[0:i]): sum(arr[0:i]) + arr[i], :]
        test_labels = labels[sum(arr[0:i]): sum(arr[0:i]) + arr[i], :]

        train1_features = features[0: sum(arr[0:i]), :]
        train1_labels = labels[0: sum(arr[0:i]), :]

        train2_features = features[sum(arr[0:i]) + arr[i]: , :]
        train2_labels = labels[sum(arr[0:i]) + arr[i]:, :]

        if train1_features.shape[0] == 0:
            train_features = train2_features
            train_labels= train2_labels
        elif train2_features.shape[0] == 0:
            train_features = train1_features
            train_labels= train1_labels
        else:
            train_features = np.concatenate((train1_features,train2_features), axis=0) 
            train_labels = np.concatenate((train1_labels,train2_labels), axis=0) 



        lst.append((train_features,train_labels,test_features,test_labels))


    return lst
        
        
        
        #what if thyere different length and how does it work on remaining 



    

