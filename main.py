import os
import torch
import numpy as np
from tqdm import tqdm

from adapt import get_method
from utils import datasets 
from utils.misc import set_global_seeds, save_configuration
from argparser import argparser, add_method_specific_args
    

def main():
    # Initial argument parsing to get the method
    initial_parser = argparser()
    initial_args, _ = initial_parser.parse_known_args()

    # Create a new parser with method-specific arguments
    parser = argparser()
    parser = add_method_specific_args(parser, initial_args.method)
    args = parser.parse_args()

    # Set the global random seed for reproducibility
    set_global_seeds(args.seed)

    # Save the configuration settings
    save_configuration(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setting up the model and the method
    adapt_method = get_method(args, device)
    results_path = os.path.join(args.save_dir, "results.txt")
    
    for corruption in args.corruptions_list:
        data_loader, classes = datasets.prepare_data(args.dataset, args.data_dir, corruption=corruption, batch_size=args.batch_size, num_workers=args.workers, severity=args.severity)
        acc = []
        iter_acc_list = []
        for t in range(args.trials):
            adapt_method.reset(full=True)
            correct = 0
            for batch_idx, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
                
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # perform adaptation
                if not(args.continual):
                    adapt_method.reset()
                if args.adapt:
                    adapt_method.adapt(inputs, classes, labels, reset=not(args.continual))

                # perform evaluation
                pred = adapt_method.evaluate(inputs, classes)

                # Calculate the number of correct predictions
                correctness = pred.eq(labels.view(1, -1).expand_as(pred))  
                correct += correctness.sum().item()

            acc.append(correct / len(data_loader.dataset))
            iter_acc_list.append(adapt_method.iter_acc / len(data_loader.dataset))
            print(correct / len(data_loader.dataset))
            
            
        iter_acc = torch.stack(iter_acc_list, dim=0)
        print(str(round(np.array(acc).mean()*100, 2)) + ',' + str(round(np.array(acc).std()*100, 2)))    
            
        with open(results_path, 'a') as fichier:
            fichier.write(str(iter_acc.shape[1]) + ',' + corruption + ',' + str(round(np.array(acc).mean()*100, 2)) + ',' + str(round(np.array(acc).std()*100, 2)) + ',' + '\n')
        detailed_results_file = "detailed_results_" + corruption + ".txt"
        with open(os.path.join(args.save_dir, detailed_results_file), 'a') as fichier:
            for i in range(iter_acc.shape[1]):
                fichier.write(f'{i},{iter_acc.mean(dim=0)[i]},{iter_acc.std(dim=0)[i]}\n')
            
if __name__ == "__main__":
    main()
