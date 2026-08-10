import argparse

def argparser():
    parser = argparse.ArgumentParser("Test-Time Adaptation of CLIP")

    # Directories
    parser.add_argument('--data_dir', type=str, default='', help='Root directory for datasets')
    parser.add_argument('--save_dir', type=str, default='save/', help='Path for saving base training weights and results')

    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Model
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='Model backbone to use') 

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'tiny-imagenet', 'caltech101', 'dtd', 'eurosat', 'fgvc_aircraft', 'flower102', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'ucf101'), help='Dataset to use')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loading')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--trials', default=3, type=int, help='Number of trials to repeat the experiments')

    # Evaluation settings
    parser.add_argument('--adapt', action='store_true', help='Enable adaptation')
    parser.add_argument('--continual', action='store_true', help='Enable continual adaptation instead batch-wise adaptation')

    # Corruptions settings
    parser.add_argument('--corruptions_list', nargs='+', default=None, type=str, help='List of corruptions to apply to the dataset (corruption datasets)')

    # Method name
    parser.add_argument('--method', type=str, default='ttsgm', choices=('ttsgm'), help='Method to use for adaptation')

    return parser

def add_method_specific_args(parser, method):
    '''
    Add method-specific arguments to the parser
    '''

    if method == 'ttsgm':
        parser.add_argument('--sparsity', type=float, default=0.999, help='Sparsity for Sparse Optimization')
        parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for Sparse Optimization')
        parser.add_argument('--watt_temps', type=str, default='templates.yaml', help='Path to the templates.yaml file')
        parser.add_argument('--use_single_template', action='store_true', help='Use REFERENCE_TEMPLATE during adaptation instead of averaging text embeddings of different templates')
        parser.add_argument('--encoder', type=str, default='vision', help='Encoder to use for Sparse Optimization')
        parser.add_argument('--sparse_mode', type=str, default='topk', choices=('topk', 'random'), help='criterion for parameter selection')
        parser.add_argument('--mask_refresh', type=int, default=0, help='Number of iterations before the sparsity mask is recomputed. If set to 0, the mask is kept static')
        parser.add_argument('--multi_template_mask_method', type=str, default='mean_abs', choices=('mean', 'mean_abs', 'vote', 'variance', 'union', 'intersection'))
        parser.add_argument('--multi_template_sign_method', type=str, default='mean_sign', choices=('vote_sign', 'mean_sign', 'mean'))
        parser.add_argument('--no_normalize_grads', action='store_true', help='Don\'t use grad normalization for multi-template gradient methods')
        parser.add_argument('--nb_templates', type=int, default=8, help='nb of tempaltes to use for MT accumulation')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return parser