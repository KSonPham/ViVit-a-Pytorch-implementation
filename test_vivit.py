import argparse
import torch
from tqdm import tqdm

from utils.data_utils import get_loader
import numpy as np
import torch.nn.functional as F
from models.modeling import MyViViT
from os import listdir
from os.path import isfile, join
from pandas import DataFrame
def test(args):
    # Test
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        config = checkpoint["config"]
        model = MyViViT(config, config.img_size, config.num_classes, config.num_frames)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(args.device)
        args.img_size = config.img_size
        args.num_frames = config.num_frames

    _, test_loader = get_loader(args)
    model.eval()
    
    all_preds = [] 
    
    epoch_iterator = tqdm(test_loader,
                          desc="Testing...",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    test_iter = args.test_iteration
    livenessscore = []


    for i in range(test_iter):
        liveness = None
        for step, batch in enumerate(epoch_iterator):
            batch = batch.to(args.device)
            x = batch
            # batch = batch.to(args.device)
            # x = batch
            with torch.no_grad():
                logits = model(x)
                score = F.softmax(logits,dim=1)[:,1]
                if liveness != None:
                    liveness = torch.cat((liveness,score), dim=0)
                else:
                    liveness = score

        livenessscore.append(liveness)
    offical_score = sum(livenessscore)/test_iter
    offical_score = offical_score.tolist()
    videos = sorted(listdir(args.test_dir  + "/videos"))
    temp = zip(videos,offical_score)
    out = DataFrame(temp, columns=['fname', 'liveness_score'])
    out.to_csv(join("results", args.name + "_" + args.test_dir +".csv"), sep = '\t',index=False)
    return 

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=False, default="test",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100","custom"], default="custom",
                        help="Which downstream task.")
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--checkpoint", type=str, default="output/base_small_checkpoint.bin",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--test_dir", default="public", type=str,
                        help="Path to the test data.")
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Path to the data.")
    

    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--test_iteration", default=5, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    
    
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
   
    test(args)
   

if __name__ == "__main__":
    main()