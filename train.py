import argparse
import time
import torch
import torch.nn as nn
from preprocess_data import PickleDataLoad
from model import TCN
from torch import optim
from matplotlib import pyplot as plt
from pathlib import Path
from anomalyDetector import fit_norm_distribution_param
import os

parser = argparse.ArgumentParser(description='PyTorch Temporal Conv Net model on Time-series dataset')
parser.add_argument('--data', type=str, default='nyc_taxi',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='nyc_taxi.pkl')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--clip', type=float, default=10)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=128)
parser.add_argument('--window_size', type=int, default=400, help='input window size')
parser.add_argument('--val_window_size', type=int, default=100, help='last length used for calculating loss')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--pred_window_size', type=int, default=1, help='predict window size,shift length for target seq')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='report interval')
parser.add_argument('--save_interval', type=int, default=5, metavar='N', help='save interval')
parser.add_argument('--save_fig', action='store_true', default=True, help='save figure')
parser.add_argument('--resume', '-r', type=bool, default=False,
                    help='use checkpoint model parameters as initial parameters (default: False)')
parser.add_argument('--pretrained', '-p', type=bool, default=False,
                    help='use checkpoint model parameters and do not train anymore')

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

##################################################################################
# load data
##################################################################################
Data = PickleDataLoad(data_type=args.data, filename=args.filename, window_size=args.window_size + args.pred_window_size)
# print(Data.testLabel.mean())  # print anomaly rate
train_data = Data.batchify(args, Data.trainData, args.batch_size)
test_data = Data.batchify(args, Data.testData, args.eval_batch_size, isLabel=False)
# test_label = Data.batchify(args, Data.testLabel, args.eval_batch_size, isLabel=True)
##################################################################################
# build model
##################################################################################
feature_dim = Data.trainData.size(1)
model = TCN(input_size=feature_dim, output_size=feature_dim, num_channels=[32, 32, 32, 32, 32, 32, 32], kernel_size=7,
            dropout=0.2).to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.MSELoss()


##################################################################################
# training
##################################################################################
def train(args, model, train_dataset, epoch):
    with torch.enable_grad():
        model.train()
        total_loss = 0.0
        start_time = time.time()
        for i in range(train_dataset.size(0)):
            # input [:window_size], target [-val_window_size:]
            input, y = train_dataset[i, :, :, :args.window_size], train_dataset[i, :, :, -args.val_window_size:]
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output[:, :, -args.val_window_size:], y)
            loss.backward()

            # clip grad
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()

            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.4f} | loss {:5.4f}'.
                      format(epoch, i, len(train_dataset), elapsed * 1000 / args.log_interval, cur_loss))
                total_loss = 0.0
                start_time = time.time()


def evaluate(args, model, test_dataset):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i in range(test_dataset.size(0)):
            input, y = test_dataset[i, :, :, :args.window_size], test_dataset[i, :, :, -args.val_window_size:]
            output = model(input)
            loss = criterion(output[:, :, -args.val_window_size:], y)

            total_loss += loss.item()

    return total_loss / (i + 1)


# loop over epochs
checkpoint_path = Path('save', args.data, 'checkpoint')
print(checkpoint_path)
print(os.getcwd())
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

if args.resume or args.pretrained:
    print('=> loading checkpoint')
    checkpoint = torch.load(Path(checkpoint_path, args.filename).with_suffix('.pth'))
    args, start_epoch, best_val_loss = checkpoint['args'], checkpoint['epoch'], checkpoint['best_loss']
    optimizer.load_state_dict((checkpoint['optimizer']))
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    epoch = start_epoch
    print('=> loaded checkpoint')
else:
    epoch = 1
    start_epoch = 1
    best_val_loss = 10
    print("=> Start training from scratch")
print('-' * 89)
print(args)
print('-' * 89)

if not args.pretrained:
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            train(args, model, train_data, epoch)
            val_loss = evaluate(args, model, test_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.
                  format(epoch, (time.time() - epoch_start_time), val_loss))
            print('-' * 89)

            if epoch % args.save_interval == 0:
                # Save the model if the validation loss is the best we've seen so far.
                is_best = val_loss < best_val_loss
                best_val_loss = min(val_loss, best_val_loss)
                model_dictionary = {'epoch': epoch,
                                    'best_loss': best_val_loss,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'args': args
                                    }
                if is_best:
                    torch.save(model_dictionary, Path(checkpoint_path, args.filename).with_suffix('.pth'))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

# calculate mean and covariance for each channel's prediction errors,and save them with the trained model
print('=> calculating mean and corvariance')
mean, cov = fit_norm_distribution_param(args, model, train_data, feature_dim)
# print('mean:{:5.4f}, cov:{:5.4f}'.format(mean, cov))
print('mean:', mean)
print('cov:', cov)
model_dictionary = {'epoch': max(epoch, start_epoch),
                    'best_loss': best_val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    'mean': mean,
                    'cov': cov
                    }
torch.save(model_dictionary, Path(checkpoint_path, args.filename).with_suffix('.pth'))
print('-' * 89)
