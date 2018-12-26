import torch
import numpy as np


def fit_norm_distribution_param(args, model, train_data, feature_dim):
    '''
    :param args:
    :param model:
    :param train_data:  [N,bz,feature_dim,window_size]
    :param feature_dim:
    :return:
    '''
    N = (train_data.size(0)) * train_data.size(1)
    bz = train_data.size(1)
    predictions = torch.zeros([N, feature_dim]).cuda()  # [N,feature_dim]
    errors = torch.zeros([N, feature_dim]).cuda()
    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()
        for i in range(train_data.size(0)):
            # if i % 10 == 0:
            #     print(i)
            input, y = train_data[i, :, :, :-1], train_data[i, :, :, -1]
            output = model(input)  # [bz,feature_dim,window_size-1]
            predictions[i * bz:(i + 1) * bz] = output[:, :, -1]
            errors[i * bz:(i + 1) * bz] = output[:, :, -1] - y

    mean = errors.mean(dim=0)
    mul = (errors - mean).t()
    cov = mul.mm(mul.t()) / N
    # cov = np.matmul(errors - mean, (errors - mean).transpose()) / errors.shape[0]

    return mean, cov


def anomalyScore(args, model, dataset, mean, cov, feature_dim):
    '''
    :param args:
    :param model:
    :param dataset: [n,bz,feature_dim,window_size] test
    :param mean:
    :param cov:
    :return: scores [N]
    '''
    N = (dataset.size(0)) * dataset.size(1)
    bz = dataset.size(1)
    predictions = torch.zeros([N, feature_dim]).cuda()  # [N,feature_dim]
    errors = torch.zeros([N, feature_dim]).cuda()
    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()
        for i in range(dataset.size(0)):
            input, y = dataset[i, :, :, :-args.pred_window_size], dataset[i, :, :, -args.pred_window_size]
            output = model(input)
            predictions[i * bz:(i + 1) * bz] = output[:, :, -1]
            errors[i * bz:(i + 1) * bz] = output[:, :, -1] - y

    scores = torch.zeros(N)
    for i, error in enumerate(errors):
        mult1 = error - mean.unsqueeze(0)  # [1,feature_dim]
        mult2 = torch.inverse(cov)
        mult3 = mult1.t()
        score = torch.mm(mult1, torch.mm(mult2, mult3))
        scores[i] = score[0]

    return scores


def get_precision_recall(args, scores, label, num_samples, beta=1.0, sampling='log'):
    '''

    :param args:
    :param scores:
    :param label:
    :param num_samples:
    :param beta:
    :return: precision,recall,f1
    '''

    maximun = scores.max()
    if sampling == 'log':
        # Sample thresholds logarithmically
        # The sampled thresholds are logarithmically spaced between: math:`10 ^ {start}` and: math:`10 ^ {end}`.
        th = torch.logspace(0, torch.log10(maximun), num_samples).to(args.device)
    else:
        # Sample thresholds equally
        # The sampled thresholds are equally spaced points between: attr:`start` and: attr:`end`
        th = torch.linspace(0, maximun, num_samples).to(args.device)

    precision = []
    recall = []
    scores = scores.cuda()

    for i in range(len(th)):
        anomaly = (scores > th[i]).float()
        idx = anomaly * 2 + label.squeeze(1)
        tn = (idx == 0.0).sum().item()
        fn = (idx == 1.0).sum().item()
        fp = (idx == 2.0).sum().item()
        tp = (idx == 3.0).sum().item()

        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)

        if p != 0 and r != 0:
            precision.append(p)
            recall.append(r)
    precision = torch.FloatTensor(precision)
    recall = torch.FloatTensor(recall)

    f1 = (1 + beta ** 2) * (precision * recall).div(beta ** 2 * precision + recall + 1e-7)

    return precision, recall, f1
