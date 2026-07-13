import os
import torch
#import Training
import Training
import Testing
from Evaluation import main
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33222', type=str, help='init_method')
    parser.add_argument('--data_root', default='', type=str, help='data path')
    parser.add_argument('--train_steps', default=60000, type=int, help='train_steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='../pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=3e-4, type=int, help='learning rate')
    parser.add_argument('--lr_power', default=0.9, type=float, help='learning rate power')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--warm_up_epoch', default=8, type=int, help='warm up epoch')
    parser.add_argument('--checkpoint_start_epoch', default=80, type=int, help='start save checkpoint')
    parser.add_argument('--checkpoint_step', default=3, type=int, help='save checkpoint every k epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=60000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=60000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='../data/USOD10k/TR', type=str, help='Trainging set')
    parser.add_argument('--validset', default='../data/USOD10k/VAL', type=str, help='Validation set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint.pth')

    # test
    parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='../data/USOD10k/TE')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='USOD10K', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_gpus = torch.cuda.device_count()
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
    if args.Testing:
        Testing.test_net(args)
    if args.Evaluation:
        main.evaluate(args)