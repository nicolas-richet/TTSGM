import torch
import torch.nn.functional as F
import clip
import copy
from utils.misc import load_templates_from_yaml
from argparse import Namespace
import torch.nn as nn
from adapt.optimizers.MTMSparseSignSGD import MTMSparseSignSGD

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': list(range(12))
}

INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': list(range(12))
    },
    
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    
    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


REFERENCE_TEMPLATE = 'a photo of a {}'

class TTSGM:
    """
    TTSGM for CLIP model.
    This class implements the Sparse Gradient Merging method for CLIP TTA.
    """
    
    def __init__(self, model, lr, iterations, sparsity, multi_template_mask_method, multi_template_sign_method, temps_dir='templates.yaml', device = 'cpu', ref_eval = False, encoder='both', position='all', use_single_template=False, mask_refresh=0, add_ln=False, sparse_mode='topk', nb_templates=8, normalize_grads=True):
        """
        Initialize the LoRA class.
        
        Args:
            model (torch.nn.Module): The CLIP model to adapt.
            lr (float): Learning rate for the optimizer.
            iterations (int): Number of iterations for training.
            device (str): Device to use for training ('cpu' or 'cuda').
        """
        base_model, _ = clip.load(model, device)
        print(f"model device: {device}")
        self.model = base_model
        self.lr = lr
        self.iterations = iterations
        self.device = device
        self.ref_eval = ref_eval or use_single_template
        self.use_single_template = use_single_template        
        self.iter_acc = torch.zeros(iterations)

        self.mask_refresh = mask_refresh
        self.add_ln =add_ln
        self.sparse_mode = sparse_mode
        self.multi_template_mask_method = multi_template_mask_method
        self.multi_template_sign_method = multi_template_sign_method
        self.nb_templates = nb_templates
        self.normalize_grads = normalize_grads
        
        # Load the text templates
        self.all_templates = load_templates_from_yaml(temps_dir)
        
        so_args = Namespace()
        so_args.encoder = encoder
        so_args.position = position
        so_args.backbone = model
        self.model = self.model.to(self.device)
        mark_as_trainable(self.model, so_args)
        density_ratio = 1 - sparsity
        self.density_ratio = density_ratio
        params = [
            {'params': [param for param in self.model.parameters() if param.requires_grad], 'dense': False},
        ]
        
        if self.add_ln:
            # adding ln grads
            add_ln_grads(self.model)
            params.append({'params': [param for name, param in self.model.named_parameters() if 'ln' in name], 'dense': True})

        if mask_refresh > 0:
            self.optimizer = MTMSparseSignSGD(params=params, lr=lr,
                                              mask_method=self.multi_template_mask_method, sign_method=self.multi_template_sign_method,
                                              density_ratio=density_ratio, static_mask=False,
                                              mask_refresh=mask_refresh, sparse_mode=self.sparse_mode, normalize_grads=self.normalize_grads)
        else:
            self.optimizer = MTMSparseSignSGD(params=params, lr=lr,
                                              mask_method=self.multi_template_mask_method, sign_method=self.multi_template_sign_method,
                                              density_ratio=density_ratio, sparse_mode=self.sparse_mode, normalize_grads=self.normalize_grads)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
        
        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)
        
    def adapt(self, x, classes, labels, reset=True):
        """Forward pass with LoRA adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.
            labels: Ground truth labels for the input images.
        """
        
        if reset:
            self.reset()
        self.perform_adaptation(x, classes, labels)
        
    @torch.no_grad()
    def evaluate(self, x, classes):
        """
        Forward pass without adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        Returns:
            pred: Predicted class labels for the input images.

        """
        
        
        # extracting features
        image_features = self.model.encode_image(x)
        
        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if self.ref_eval:
            text_features = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=True)
        else:
            text_features = self.extract_text_embeddings(classes, self.all_templates, average=True)
        text_features = text_features.T

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, pred = similarity.topk(1, 1, True, True)
        pred = pred.t()
       
        return pred
    
    def reset(self, full = False):
        """
        Resets the model and optimizer to their initial states.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)
        
        params = [
            {'params': [param for name, param in self.model.named_parameters() if (param.requires_grad and not ('ln' in name))], 'dense': False},
        ]
        if self.add_ln:
            # adding ln grads
            params.append({'params': [param for name, param in self.model.named_parameters() if 'ln' in name], 'dense': True})

        if self.mask_refresh > 0:
            self.optimizer = MTMSparseSignSGD(params=params, lr=self.lr,
                                              mask_method=self.multi_template_mask_method, sign_method=self.multi_template_sign_method,
                                              density_ratio=self.density_ratio, static_mask=False,
                                              mask_refresh=self.mask_refresh, sparse_mode=self.sparse_mode, normalize_grads=self.normalize_grads)
        else:
            self.optimizer = MTMSparseSignSGD(params=params, lr=self.lr,
                                              mask_method=self.multi_template_mask_method, sign_method=self.multi_template_sign_method,
                                              density_ratio=self.density_ratio, sparse_mode=self.sparse_mode, normalize_grads=self.normalize_grads)
        if full:
            self.iter_acc = torch.zeros(self.iterations)

        
    def perform_adaptation(self, x, classes, labels):
        """
        Perform adaptation on the input data.

        Args:
            x: Input image tensor.
            classes: List of class names.
            labels: Ground truth labels for the input images.
        """
        templates_embeddings = [self.extract_text_embeddings(classes, [template], average=True).t() for template in self.all_templates[:self.nb_templates]]
        
        for i in range(self.iterations):

            
            #if i < 5 or i%5==0:
            #    # Evaluating the model at each iteration
            #    self.model.eval()
            #    pred = self.evaluate(x, classes)
            #    correctness = pred.eq(labels.view(1, -1).expand_as(pred))
            #    self.iter_acc[i]+= correctness.sum().item()
            #self.model.train()
            
            with torch.no_grad():
                image_features = self.model.encode_image(x)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            images_similarity = image_features @ image_features.t() 
            
            self.model.train()
            self.optimizer.begin_template_accumulation()
            mean_loss = 0
            for text_x in templates_embeddings:
                self.optimizer.zero_grad()
                similarity = (100 * image_features @ text_x.t()).softmax(1)
                values, pred = similarity.topk(1, 1, True, True)
                pred_inputs = torch.cat([text_x[c,] for c in pred])
                logits, image_features, text_features = self.model(x, pred_inputs, True)
                
                texts_similarity = pred_inputs @ pred_inputs.t()
                
                targets = F.softmax(((images_similarity + texts_similarity) / 2) / 0.01, dim=-1)
                loss = self.cross_entropy(logits.t(), targets.t(), reduction='mean')
                loss.backward()
                mean_loss += loss.item()
                self.optimizer.accumulate_template()
            self.optimizer.finalize()
            self.optimizer.step()
            self.optimizer.zero_grad()    
            
            #if i < 5 or i%5==0:  
            #    print(f"Iteration {i}/{self.iterations}, Accuracy: {correctness.sum().item() / len(labels)}")
            #    print(f"Loss: {mean_loss}")
            
            if self.scheduler is not None:
                self.scheduler.step()
                
                
            
    
    def extract_text_embeddings(self, class_names, templates, average=True):  
        """
        Extracts text embeddings for given class names and templates.

        Args:
            class_names: List of class names to generate text embeddings for.
            templates: List of text templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different templates for each class.

        Returns:
            text_features: Tensor of text embeddings for the given class names and templates.
        """
        text_features = []
        for class_name in class_names:
            texts = [template.format(class_name) for template in templates]
            texts = clip.tokenize(texts).to(self.device) 
            class_embeddings = self.model.encode_text(texts) 
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm()
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features
    
    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """
        Copy the model and optimizer states for resetting after adaptation.

        Args:
            model: The model to copy.
            optimizer: The optimizer to copy.

        Returns:
            model_state: Copied state of the model.
            optimizer_state: Copied state of the optimizer.
        """
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state


    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """
        Restore the model and optimizer states from copies.

        Args:
            model: The model to restore.
            optimizer: The optimizer to restore.
            model_state: The state to restore the model to.
            optimizer_state: The state to restore the optimizer to.
        """
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
    
    @staticmethod
    def cross_entropy(preds, targets, reduction='none'):
        """
        Calculate the cross-entropy loss between predictions and targets.

        Args:
            preds: Predicted logits.
            targets: Target probabilities.
            reduction: Type of reduction to apply to the output ('none' or 'mean').

        Returns:
            The computed loss.
        """
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
          
def mark_as_trainable(model, args):
    for _, param in model.named_parameters():
        param.requires_grad = False
    trainable_params = [] 
    if args.encoder in ['text', 'both']:
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = model.transformer  
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, param in block.named_parameters():
                    if "attn" in name and "weight" in name:  
                        param.requires_grad = True  
                        trainable_params.append((f"text_encoder.resblocks[{i}].{name}", tuple(param.shape)))
                    elif "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name: 
                        param.requires_grad = True  
                        trainable_params.append((f"text_encoder.resblocks[{i}].{name}", tuple(param.shape)))
    if args.encoder in ['vision', 'both']:
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, param in block.named_parameters():
                    if "attn" in name and "weight" in name: 
                        param.requires_grad = True  
                        trainable_params.append((f"vision_encoder.resblocks[{i}].{name}", tuple(param.shape)))
                    elif "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name: 
                        param.requires_grad = True  
                        trainable_params.append((f"vision_encoder.resblocks[{i}].{name}", tuple(param.shape)))
    print("\n✅ Trainable Parameters in CLIP-SRGIM:")
    for param_name, shape in trainable_params:
        print(f"{param_name}: {shape} ({param.dtype})")
        


def add_ln_grads(model):
    """
    Set gradient settings for LayerNorm layers within the model, disabling gradients globally except for these LN layers.

    Args:
        model: The model whose LayerNorm layers' gradients are to be set.
    
    Returns:
        The model with modified gradient settings.
    """
    trainable_params = []
    vision_encoder = model.visual.transformer
    for i, block in enumerate(vision_encoder.resblocks):
                for name, param in block.named_parameters():
                    print(name)
                    if "ln" in name: 
                        param.requires_grad = True  
                        trainable_params.append((f"vision_encoder.resblocks[{i}].{name}", tuple(param.shape)))