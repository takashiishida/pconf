"""A simple demo to perform synthetic experiments for Pconf classification."""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from sklearn import svm

def getPositivePosterior(x, mu1, mu2, cov1, cov2, positive_prior):
    """Returns the positive posterior p(y=+1|x)."""
    conditional_positive = np.exp(-0.5 * (x - mu1).T.dot(np.linalg.inv(cov1)).dot(x - mu1)) / np.sqrt(np.linalg.det(cov1)*(2 * np.pi)**x.shape[0])
    conditional_negative = np.exp(-0.5 * (x - mu2).T.dot(np.linalg.inv(cov2)).dot(x - mu2)) / np.sqrt(np.linalg.det(cov2)*(2 * np.pi)**x.shape[0])
    marginal_dist = positive_prior * conditional_positive + (1 - positive_prior) * conditional_negative
    positivePosterior = conditional_positive * positive_prior / marginal_dist
    return positivePosterior

class LinearNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def getAccuracy(x_test, y_test, model):
    """Calculates the classification accuracy."""
    predicted = model(Variable(torch.from_numpy(x_test)))
    accuracy = np.sum(torch.sign(predicted).data.numpy() == np.matrix(y_test).T) * 1. / len(y_test)
    return accuracy

def pconfClassification(num_epochs, lr, x_train_p, x_test, y_test, r):
    model = LinearNetwork(input_size=2, output_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(x_train_p))
        confidence = Variable(torch.from_numpy(r))
        optimizer.zero_grad()
        negative_logistic = nn.LogSigmoid()
        logistic = -1. * negative_logistic(-1. * model(inputs))
        loss = torch.sum(-model(inputs)+logistic * 1. / confidence)  # note that \ell_L(g) - \ell_L(-g) = -g with logistic loss
        loss.backward()
        optimizer.step()
    params = list(model.parameters())
    accuracy = getAccuracy(x_test=x_test, y_test=y_test, model=model)
    return params, accuracy

def naiveClassification(num_epochs, lr, x_naive, y_naive, y_test, x_test, R):
    model = LinearNetwork(input_size=2, output_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(x_naive))
        targets = Variable(torch.from_numpy(y_naive))
        confidence = Variable(torch.from_numpy(R))
        optimizer.zero_grad()
        negative_logistic = nn.LogSigmoid()
        logistic = -1. * negative_logistic(targets * model(inputs))
        loss = torch.sum(logistic * confidence)
        loss.backward()
        optimizer.step()
    params = list(model.parameters())
    accuracy = getAccuracy(x_test=x_test, y_test=y_test, model=model)
    return params, accuracy

def osvmClassification(nu, x_train_p, x_test, y_train, y_test):
    clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=0.1)
    clf.fit(x_train_p)
    y_pred = clf.predict(x_test)
    accuracy = np.sum(y_pred == y_test) / len(y_pred)
    return clf, accuracy

def supervisedClassification(num_epochs, lr, x_train, x_test, y_train, y_test):
    y_train_matrix = np.matrix(y_train).T.astype('float32')
    model = LinearNetwork(input_size=2, output_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(x_train))
        targets = Variable(torch.from_numpy(y_train_matrix))
        optimizer.zero_grad()
        negative_logistic = nn.LogSigmoid()
        logistic = -1. * negative_logistic(targets * model(inputs))
        loss = torch.sum(logistic)
        loss.backward()
        optimizer.step()
    params = list(model.parameters())
    accuracy = getAccuracy(x_test=x_test, y_test=y_test, model=model)
    return params, accuracy

def generateData(mu1, mu2, cov1, cov2, n_positive, n_negative, n_positive_test, n_negative_test):
    positive_prior = n_positive/(n_positive + n_negative)
    x_train_p = np.random.multivariate_normal(mu1, cov1, n_positive)
    x_train_n = np.random.multivariate_normal(mu2, cov2, n_negative)
    x_test_p = np.random.multivariate_normal(mu1, cov1, n_positive_test)
    x_test_n = np.random.multivariate_normal(mu2, cov2, n_negative_test)
    x_naive = np.r_[x_train_p, x_train_p]
    x_naive = x_naive.astype('float32')
    y_train = np.r_[np.ones(n_positive), -np.ones(n_negative)]
    y_test = np.r_[np.ones(n_positive_test), -np.ones(n_negative_test)]
    y_naive = np.r_[np.ones(n_positive), -np.ones(n_positive)]
    y_naive = np.matrix(y_naive).T.astype('float32')
    x_train = np.r_[x_train_p, x_train_n]
    x_train = x_train.astype('float32')
    x_test = np.r_[x_test_p, x_test_n]
    x_test = x_test.astype('float32')
    x_train_p = x_train_p.astype('float32')
    # calculating the exact positive-confidence values: r
    r = np.zeros(n_positive)
    for i in range(n_positive):
        x = x_train_p[i, :]
        r[i] = getPositivePosterior(x, mu1, mu2, cov1, cov2, positive_prior)
    R = np.r_[r, 1-r]
    r = np.matrix(r).T
    r = r.astype('float32')
    R = np.matrix(R).T.astype('float32')

    return r, R, x_naive, x_train, x_train_p, x_test, y_naive, y_train, y_test
