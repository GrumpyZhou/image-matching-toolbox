import os
import yaml
import immatch

def parse_model_config(config, benchmark_name, root_dir='.'):
    config_file = f'{root_dir}/configs/{config}.yml'
    with open(config_file, 'r') as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)[benchmark_name]
        
        # Update pretrained model path
        if 'ckpt' in model_conf and root_dir != '.':
            model_conf['ckpt'] = os.path.join(root_dir, model_conf['ckpt'])
            if 'coarse' in model_conf and 'ckpt' in model_conf['coarse']:
                model_conf['coarse']['ckpt'] = os.path.join(
                    root_dir, model_conf['coarse']['ckpt']
                )
    return model_conf

def init_model(config, benchmark_name, root_dir='.'):
    # Load model config
    model_conf = parse_model_config(config, benchmark_name, root_dir)

    # Initialize model
    class_name = model_conf['class']
    model = immatch.__dict__[class_name](model_conf)
    print(f'Method:{class_name} Conf: {model_conf}')
    return model, model_conf
