"""
Official implementation of MCFlow -
"""
import numpy as np
import torch
import argparse
import sys
import os
from models import InterpRealNVP
import util
from loader import DataLoader
from models import LatentToLatentApprox


def main():
    # initialize train dataset class
    ldr_train = DataLoader(mode=0, seed=args.seed, path="physionet_train", drp_percent=args.drp_impt)
    train_data_loader = torch.utils.data.DataLoader(ldr_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    num_neurons = int(ldr_train.train[0].shape[0])

    #initialize test dataset class
    ldr_test = DataLoader(mode=0, seed=args.seed, path="physionet_test", drp_percent=0)
    test_data_loader = torch.utils.data.DataLoader(ldr_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Initialize normalizing flow model neural network and its optimizer
    flow = util.init_flow_model(num_neurons, args.num_nf_layers, InterpRealNVP, ldr_train.train[0].shape[0], args)
    nf_optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad], lr=args.lr)

    # Initialize latent space neural network and its optimizer
    num_hidden_neurons = [int(ldr_train.train[0].shape[0]), int(ldr_train.train[0].shape[0]), int(ldr_train.train[0].shape[0]),
                          int(ldr_train.train[0].shape[0]), int(ldr_train.train[0].shape[0])]
    nn_model = LatentToLatentApprox(int(ldr_train.train[0].shape[0]), num_hidden_neurons).float()
    if args.use_cuda:
        nn_model.cuda()
    nn_optimizer = torch.optim.Adam([p for p in nn_model.parameters() if p.requires_grad], lr=args.lr)

    reset_scheduler = 2

    print("\n****************************************")
    print("Staring Approx-comp experiment")

    # Train and test MCFlow
    for epoch in range(args.n_epochs):
        util.endtoend_train(flow, nn_model, nf_optimizer, nn_optimizer, train_data_loader, args)  # Train the MCFlow model

        with torch.no_grad():
            ldr_test.mode = 1  # Use testing data
            te_mse, _ = util.endtoend_test(flow, nn_model, test_data_loader, args)  # Test MCFlow model
            ldr_train.mode = 0  # Use training data
            print("Epoch", epoch, " Test RMSE", te_mse ** .5)

        if (epoch + 1) % reset_scheduler == 0:
            # Reset unknown values in the dataset using predicted estimates
            ldr_train.reset_imputed_values(nn_model, flow, args.seed, args)
            # Initialize brand new flow model to train on new dataset
            flow = util.init_flow_model(num_neurons, args.num_nf_layers, InterpRealNVP, ldr.train[0].shape[0], args)
            nf_optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad], lr=args.lr)
            reset_scheduler = reset_scheduler * 2


''' Run MCFlow experiment '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Reproducibility')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-nf-layers', type=int, default=3)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--drp-impt', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use-cuda', type=util.str2bool, default=True)
    args = parser.parse_args()

    ''' Reproducibility '''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ''' Cuda enabled experimentation check '''
    if not torch.cuda.is_available() or not args.use_cuda:
        print("CUDA Unavailable. Using cpu. Check torch.cuda.is_available()")
        args.use_cuda = False

    if not os.path.exists('masks'):
        os.makedirs('masks')

    if not os.path.exists('data'):
        os.makedirs('data')

    main()
    print("Experiment completed")
