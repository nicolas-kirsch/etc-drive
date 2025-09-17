import argparse, math


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Mechanical experiment.")

    # experiment
    parser.add_argument('--random-seed', type=int, default=1, help='Random seed. Default is 5.')

    # stiffness
    parser.add_argument('--lower-stiff', type=bool, default=False, help='Make the system less stiff to speed computation')

    # dataset
    parser.add_argument('--horizon', type=int, default=int(1e+4), help='Time horizon for the computation. Default is 24.')
    parser.add_argument('--train-horizon', type=int, default=int(1e+4), help='Time horizon for the computation. Default is 24.')

    parser.add_argument('--state-dim', type=int, default=1, help='Number of states of the LTI Plant. Default is 1.')
    parser.add_argument('--num-rollouts', type=int, default=10, help='Number of rollouts in the training data. Default is 30.')
    parser.add_argument('--phase-loss', type=int, default=2, help='How many phases are lost in the fault. Default is 2.')

    # optimizer
    parser.add_argument('--batch-size', type=int, default=15, help='Number of forward trajectories of the closed-loop system at each step. Default is 5.')
    parser.add_argument('--epochs', type=int, default=10, help='Total number of epochs for training. Default is 5000 if collision avoidance, else 100.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default is 2e-3 if collision avoidance, else 5e-3.')
    parser.add_argument('--log-epoch', type=int, default=-1, help='Frequency of logging in epochs. Default is 0.1 * epochs.')
    parser.add_argument('--return-best', type=bool, default=True, help='Return the best model on the validation data among all logged iterations. The train data can be used instead of validation data. The Default is True.')


    # controller
    parser.add_argument('--cont-init-std', type=float, default=0.1, help='Initialization std for controller params. Default is 0.1.')
    parser.add_argument('--dim-internal', type=int, default=10, help='Dimension of the internal state of the controller. Adjusts the size of the linear part of REN. Default is 10.')
    parser.add_argument('--l', type=int, default=10, help='size of the non-linear part of REN. Default is 8.')   # default was 4

    # loss
    parser.add_argument('--alpha-v-max', type=float, default=0, help='Weight of the loss due to control input "u". Default is 0.1/400.')  #TODO: 400 is output_amplification^2
    parser.add_argument('--Q', type=float, default=1, help='Weight of the loss due to control input "u". Default is 0.1/400.')  #TODO: 400 is output_amplification^2
    parser.add_argument('--alpha-col', type=float, default=100, help='Weight of the collision avoidance loss. Default is 100 if "col-av" is True, else None.')
    parser.add_argument('--alpha-obst', type=float, default=5e3, help='Weight of the obstacle avoidance loss. Default is 5e3 if "obst-av" is True, else None.')
    parser.add_argument('--min-dist', type=float, default=1.0, help='TODO. Default is 1.0 if "col-av" is True, else None.')  #TODO: add help


    

    args = parser.parse_args()


    if args.log_epoch == -1 or args.log_epoch is None:
        args.log_epoch = math.ceil(float(args.epochs)/10)


    return args


def print_args(args):
    msg = '\n[INFO] Dataset:  -- num_rollouts: %i' % args.num_rollouts
    msg += ' -- test horizon: %i' % args.horizon + ' -- train horizon: %i' % args.train_horizon

    msg += '\n[INFO] FAULT TYPE: phase lost: %i' % args.phase_loss 

    msg += '\n[INFO] Controller: dimension of the internal state: %i' % args.dim_internal
    msg += ' -- dim_nl: %i' % args.l + ' -- cont_init_std: %.2f'% args.cont_init_std


    msg += '\n[INFO] Optimizer: lr: %.2e' % args.lr
    msg += ' -- batch_size: %i' % args.batch_size + ', -- return best model for validation data among logged epochs:' + str(args.return_best)

    return msg

