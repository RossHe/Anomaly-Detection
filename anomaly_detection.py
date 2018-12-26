import torch
import torch.nn as nn
from preprocess_data import PickleDataLoad
from model import TCN
from torch import optim
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import argparse
from anomalyDetector import *

parser = argparse.ArgumentParser(description='Pytorch TCN Anomaly Detection Model')
parser.add_argument('--data', type=str, default='ecg',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='chfdb_chf01_275.pkl')
parser.add_argument('--beta', type=float, default=1.0, help='beta value for f-beta score')
parser.add_argument('--save_fig', type=bool, default=True)

args_ = parser.parse_args()
print('-' * 89)
print("=> loading checkpoint ")
checkpoint = torch.load(str(Path('save', args_.data, 'checkpoint', args_.filename).with_suffix('.pth')))
args = checkpoint['args']
args.beta = args_.beta
args.save_fig = args_.save_fig
print('=> loaded checkpoint')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
Data = PickleDataLoad(data_type=args.data, filename=args.filename, window_size=args.window_size + args.pred_window_size,
                      augment_test_data=False)
# print(Data.testLabel.mean())  # print anomaly rate
train_data = Data.batchify(args, Data.trainData, args.batch_size)
test_data = Data.batchify(args, Data.testData, args.eval_batch_size, isLabel=False)
test_label = Data.batchify(args, Data.testLabel, args.eval_batch_size, isLabel=True)
print('anomaly rate:', Data.testLabel.mean())  # print anomaly rate

###############################################################################
# Build the model
###############################################################################
feature_dim = Data.trainData.size(1)
model = TCN(input_size=feature_dim, output_size=feature_dim, num_channels=[32, 32, 32, 32, 32, 32, 32],
            kernel_size=7, dropout=0.2).to(args.device)
model.load_state_dict(checkpoint['state_dict'])

try:
    '''load mean and covariance if they are pre-calculated, if not calculate them.'''
    # Mean and covariance are calculated on train dataset.
    if 'mean' in checkpoint.keys() and 'cov' in checkpoint.keys():
        print('=> loading pre-calculated mean and covariance')
        mean, cov = checkpoint['mean'], checkpoint['cov']
    else:
        print('=> calculating mean and covariance')
        mean, cov = fit_norm_distribution_param(args, model, train_data, feature_dim)

    '''calculate anomaly scores'''
    # Anomaly scores are calculated on the test dataset
    # given the mean and the covariance calculated on the train dataset
    print('=> calculating anomaly scores')
    scores = anomalyScore(args, model, test_data, mean, cov, feature_dim)

    '''visualize the score'''
    plt.plot(scores.numpy(), label='score')
    plt.plot(test_label.cpu().numpy(), label='label')
    plt.show()

    '''evaluate the result'''
    # The obtained anomaly scores are evaluated by measuring precision, recall, and f_beta scores
    # The precision, recall, f_beta scores are are calculated repeatedly,
    # sampling the threshold from 0 to the maximum anomaly score value, either equidistantly or logarithmically.
    precision, recall, f_beta = get_precision_recall(args, scores, test_label, num_samples=1000, beta=args.beta,
                                                     sampling='log')

    if args.save_fig:
        save_dir = Path('result', args.data, args.filename).with_suffix('').joinpath('fig_detection')
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.plot(precision.cpu().numpy(), label='precision')
        plt.plot(recall.cpu().numpy(), label='recall')
        plt.plot(f_beta.cpu().numpy(), label='f1')
        plt.legend()
        plt.xlabel('Threshold (log scale)')
        plt.ylabel('Value')
        plt.title('Anomaly Detection on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
        plt.savefig(str(save_dir.joinpath('fig_f_beta').with_suffix('.png')))
        plt.show()
        plt.close()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

print('=> saving the results as pickle extensions')
