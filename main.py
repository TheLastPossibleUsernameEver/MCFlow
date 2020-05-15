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
    # initialize dataset class
    ldr = DataLoader(mode=0, seed=args.seed, path=args.dataset, drp_percent=args.drp_impt)
    print("Initialized Data Loader")

    data_loader = torch.utils.data.DataLoader(ldr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    print("Initialized Torch Data Loader")

    num_neurons = int(ldr.train[0].shape[0])
    print("Initialized num_neurons")

    # Initialize normalizing flow model neural network and its optimizer
    flow = util.init_flow_model(num_neurons, args.num_nf_layers, InterpRealNVP, ldr.train[0].shape[0], args)
    print("Initialized normalizing flow model")
    nf_optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad], lr=args.lr)
    print("Initialized normalizing flow model neural network optimizer")

    # Initialize latent space neural network and its optimizer
    print("Initialize latent space neural network")
    num_hidden_neurons = [int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]),
                          int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0])]

    print("Initialize neural network optimizer")
    nn_model = LatentToLatentApprox(int(ldr.train[0].shape[0]), num_hidden_neurons).float()

    if args.use_cuda:
        nn_model.cuda()
    nn_optimizer = torch.optim.Adam([p for p in nn_model.parameters() if p.requires_grad], lr=args.lr)

    reset_scheduler = 2

    if args.dataset == 'physionet':
        print("\n****************************************")
        print("Starting Approx-Comp experiment\n")
    else:
        print("Invalid dataset error")
        sys.exit()

    # Train and test MCFlow
    for epoch in range(args.n_epochs):
        util.endtoend_train(flow, nn_model, nf_optimizer, nn_optimizer, data_loader, args)  # Train the MCFlow model

        with torch.no_grad():
            ldr.mode = 1  # Use testing data
            te_mse, _ = util.endtoend_test(flow, nn_model, data_loader, args)  # Test MCFlow model
            ldr.mode = 0  # Use training data
            print("Epoch", epoch, " Test RMSE", te_mse ** .5)

        if (epoch + 1) % reset_scheduler == 0:
            # Reset unknown values in the dataset using predicted estimates
            ldr.reset_imputed_values(nn_model, flow, args.seed, args)
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
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--drp-impt', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use-cuda', type=util.str2bool, default=False)
    parser.add_argument('--dataset', default='physionet', help='Two options: (1) letter-recognition or (2) mnist')
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
