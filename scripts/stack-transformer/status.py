import argparse
import re
import os


checkpoint_re = re.compile(r'checkpoint([0-9]+)\.pt')
checkpoint_best_re = re.compile(r'checkpoint_(.*best)_(.*)\.pt')
results_regex = re.compile(r'dec-checkpoint([0-9]+)\.(.+)')
config_var_regex = re.compile(r'^([^=]+)=([^ ]+).*$')


def yellow(string):
    return "\033[93m%s\033[0m" % string


def green(string):
    return "\033[92m%s\033[0m" % string


def red(string):
    return "\033[91m%s\033[0m" % string


def argument_parsing():

    # Argument hanlding
    parser = argparse.ArgumentParser(
        description='Organize model results'
    )
    # jbinfo args
    parser.add_argument(
        '--models',
        type=str,
        default='DATA/AMR/models/',
        help='Folder containing model folders (containing themselves '
             'checkpoints, config.sh etc)'
    )
    return parser.parse_args()


def get_status(models, model_name):

    # basic paths
    model_folder = f'{models}/{model_name}'
    epoch_folder = f'{model_folder}/epoch_tests/'
    
    # get needed config values
    max_epoch = None
    keep_last_epochs = 40
    with open(f'{model_folder}/config.sh') as fid:
        for line in fid.readlines():
            if config_var_regex.match(line.strip()):
                name, value = config_var_regex.match(line.strip()).groups()
                if name == 'MAX_EPOCH':
                    max_epoch = int(value)
                    break

    assert max_epoch, "Missing MAX_EPOCH in '{model_folder}/config.sh"
    
    # Get info from available files
    checkpoints = [
        int(checkpoint_re.match(m).groups()[0]) 
        for m in os.listdir(model_folder) if checkpoint_re.match(m)
    ]
    best_checkpoints = [
        checkpoint_best_re.match(m).groups()
        for m in os.listdir(model_folder) if checkpoint_best_re.match(m)
    ]
    if os.path.isdir(epoch_folder):
        results = sorted([
            int(results_regex.match(m).groups()[0]) 
            for m in os.listdir(epoch_folder) 
            if results_regex.match(m) and 
                results_regex.match(m).groups()[1] == 'actions'
        ])
    else:    
        results = []
    
    # Sanity check
    result_epochs = list(range(max_epoch - keep_last_epochs + 1, max_epoch + 1))
    if checkpoints:
    
        # pre checkpoint removal
        missing_epochs = list(set(result_epochs) - set(checkpoints))
        missing_results = list(set(result_epochs) - set(results))
        unrecoverable_results = list(set(missing_results) & set(missing_epochs))
        if results == [] and missing_epochs:
            return model_name, yellow('training'), f'epoch {max(checkpoints)}/{max_epoch}'
        elif unrecoverable_results:
            return model_name, red('broken!'), 'checkpoints deleted but missing results'
        elif missing_results:
            return model_name, 'uncompleted', f'test {len(missing_results)} models' 
        elif missing_epochs:    
            return model_name, 'uncompleted', f'train {len(missing_results)} epochs' 
        elif best_checkpoints:    
            return model_name, green('completed'), 'you could remove checkpoints' 
        else:    
            return model_name, 'uncompleted', 'run model ranker and remove checkpoints' 
    
    elif best_checkpoints:
    
        # post checkpoint removal
        missing_results = list(set(result_epochs) - set(results))
        if missing_results:
            return model_name, red('broken!'), 'checkpoints deleted but missing results'
        else:
            return model_name, green('completed'), ''
    
    else:
    
        # not trained
        return model_name, 'not trained', ''


def main():

    # ARGUMENT HANDLING
    args = argument_parsing()

    # Get status of each model
    stata = []
    for model_name in os.listdir(args.models):
        stata.append(get_status(args.models, model_name))
   
    # print
    print()
    for status in sorted(stata, key=lambda x: x[0]):
        model_name, status, detail = status
        print(f'{model_name:50s} {status} {detail}')
    print()

    

if __name__ == '__main__':
    main()
