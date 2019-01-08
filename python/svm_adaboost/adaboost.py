# Starter Code @Author Ambuj Tewari
# Full Code @Author Rui Ma
# Version 0.1

import math
import numpy as np
from svm \
    import evaluate_classifier, print_evaluation_summary


def weak_learner(instances, labels, dist):

    """ Returns the best 1-d threshold classifier.

    A 1-d threshold classifier is of the form

    lambda x: s*x[j] < threshold

    where s is +1/-1,
          j is a dimension
      and threshold is real number in [-1, 1].

    The best classifier is chosen to minimize the weighted misclassification
    error using weights from the distribution dist.

    """
    
    n = instances.shape[0]
    labels = labels.reshape(n,1)
    min = 999
    for i in range(instances.shape[1]):	           # go over each feature
        for theta in np.arange(-1.0, 1.01, 0.01):  # go over 200 thresholds
            for s in [-1, 1]: 		               # go over each sign
                errArr = np.ones((n,1))
                classifierVec = np.zeros((n,1))
                classifierVec[s * instances[:, i] < theta] = 1
                errArr[classifierVec == labels] = 0
                weighted = dist.reshape(1,n).dot(errArr).item(0)  # total err
                if weighted < min:
                    min = weighted
                    feature = i
                    thresh = theta
                    sign = s
    

    return lambda x: (sign*x[feature] < thresh).astype(int)


def compute_error(h, instances, labels, dist):

    """ Returns the weighted misclassification error of h.

    Compute weights from the supplied distribution dist.
    """
    
    n = instances.shape[0]
    labels = labels.reshape(n,1)
    err = np.array([(h(instances[i]) != labels[i]).astype(int) for \
             i in range(n)])
    weighted = dist.reshape(1,n).dot(err).item(0) 
    
    return weighted


def update_dist(h, instances, labels, dist, alpha):

    """ Implements the Adaboost distribution update. """
    
    n = instances.shape[0]
    newDist = np.zeros(n, 'float')
    for i in range(n):
        if h(instances[i]) != labels[i]:
            newDist[i] = dist[i]*np.exp(alpha)
        else:
            newDist[i] = dist[i]*np.exp(-alpha)   
    
    return newDist/np.sum(newDist) 


def run_adaboost(instances, labels, weak_learner, num_iters=20):

    n, d = instances.shape
    n1 = labels.size

    if n1 != n:
        raise Exception('Expected same number of labels as no. of rows in \
                        instances')

    alpha_h = []

    dist = np.ones(n)/n

    for i in range(num_iters):

        print "Iteration: %d" % i
        h = weak_learner(instances, labels, dist)

        error = compute_error(h, instances, labels, dist)

        if error > 0.5:
            break

        alpha = 0.5 * math.log((1-error)/error)

        dist = update_dist(h, instances, labels, dist, alpha)

        alpha_h.append((alpha, h))

    # return a classifier whose output
    # is an alpha weighted linear combination of the weak
    # classifiers in the list alpha_h
    def classifier(point):
        """ Classifies point according to a classifier combination.

        The combination is stored in alpha_h.
        """

        sum = 0.0
        for i in range(num_iters):
            sum = sum + alpha_h[i][0] * (2 * alpha_h[i][1](point) - 1)

        if sum > 0:
            return 1
        else:
            return 0


    return classifier


def main():
    data_file = 'ionosphere.data'

    data = np.genfromtxt(data_file, delimiter=',', dtype='|S10')
    instances = np.array(data[:, :-1], dtype='float')
    labels = np.array(data[:, -1] == 'g', dtype='int')

    n, d = instances.shape
    nlabels = labels.size

    if n != nlabels:
        raise Exception('Expected same no. of feature vector as no. of labels')

    train_data = instances[:200]  # first 200 examples
    train_labels = labels[:200]  # first 200 labels

    test_data = instances[200:]  # example 201 onwards
    test_labels = labels[200:]  # label 201 onwards

    print 'Running Adaboost...'
    adaboost_classifier = run_adaboost(train_data, train_labels, weak_learner)
    print 'Done with Adaboost!\n'

    confusion_mat = evaluate_classifier(adaboost_classifier, test_data,
                                        test_labels)
    print_evaluation_summary(confusion_mat)

if __name__ == '__main__':
    main()
