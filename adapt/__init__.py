from .ttsgm import TTSGM


def get_method(args, device):
     
    if args.method == "ttsgm":
        print(f"Selected method: TTSGM with parameters: backbone={args.backbone}, multi_template_sign_method={args.multi_template_sign_method}, normalize_grads={not(args.no_normalize_grads)} lr={args.lr}, iterations={args.iterations}, sparsity={args.sparsity}, nb_templates={args.nb_templates}, templates dir={args.watt_temps}, device={device}, encoder={args.encoder}, single template={args.use_single_template}, mask_refresh={args.mask_refresh}")
        return TTSGM(args.backbone, args.lr, multi_template_mask_method=args.multi_template_mask_method, multi_template_sign_method=args.multi_template_sign_method, iterations=args.iterations, sparsity=args.sparsity, device=device, temps_dir=args.watt_temps, encoder=args.encoder, use_single_template=args.use_single_template, add_ln=args.add_ln, sparse_mode=args.sparse_mode, mask_refresh=args.mask_refresh, nb_templates=args.nb_templates, normalize_grads=not(args.no_normalize_grads))
    
    else:
        raise ValueError(f"Unknown method: {args.method}")