
import boto3
import argparse

parser = argparse.ArgumentParser(description='downloading_model_aws')
parser.add_argument(
    'fpath', 
    action='store', 
    type=str, 
    help='the file path to save the downloaded model as.')
parser.add_argument(
    '--model_name',
    type=str,
    help='the name of the model you would like to download',
    default='AMR3-structbart-L'
)


MODELS = {
    'AMR3-structbart-L': 'amr3.0-structured-bart-large-neur-al-sampling5-seed42.zip'
}

def model_download(fpath, model_name='AMR3-structbart-L'):
    try:      
        model_file_name = MODELS[model_name]
    except:
        raise ValueError('the model_name entered in invalid')


    ACCESS_KEY = None
    SECRET_KEY = None
    URL = None

    s3 = boto3.client('s3', endpoint_url=URL ,aws_access_key_id=ACCESS_KEY , aws_secret_access_key=SECRET_KEY)
    s3.download_file('mnlp-models-amr','amr3.0-structured-bart-large-neur-al-sampling5-seed42.zip',fpath)

# s3.download_file('your_bucket','k.png','/Users/username/Desktop/k.png')


if __name__ == "__main__":
    args = parser.parse_args()
    model_download(args.fpth, args.model_name)