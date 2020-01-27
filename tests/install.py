"""
This is a set of usual checks for the repo installation
"""
import torch
import subprocess
from torch.utils.cpp_extension import CUDAExtension


def check_cuda_torch_binary_vs_bare_metal():

    # command line CUDA
    cuda_dir = torch.utils.cpp_extension.CUDA_HOME
    cuda_call = [cuda_dir + "/bin/nvcc", "-V"]
    raw_output = subprocess.check_output(cuda_call, universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    # torch compilation CUDA
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        print(
            "Pytorch binaries were compiled with Cuda {} but binary {} is {}".format(
                torch.version.cuda, cuda_dir + "/bin/nvcc", output[release_idx])
        )


if __name__ == '__main__':
    
    # Pytorch and CUDA
    assert torch.cuda.is_available(), "No CUDA available"
    print(f'pytorch {torch.__version__}') 
    print(f'cuda {torch.version.cuda}') 
    # happens for torch 1.2.0
    assert torch.cuda.device_count(), "0 GPUs found"
    try:
        import apex
        print("Apex installed")
    except ImportError:
        print("Apex not installed")
    check_cuda_torch_binary_vs_bare_metal()
    if torch.cuda.get_device_capability(0)[0] < 7:
        print("GPU wont support --fp")

    try:
        import torch_scatter
        print("pytorch-scatter installed")
    except ImportError:    
        print("pytorch-scatter not installed")

    try:
        import torch_scatter.scatter_cuda
    except ImportError:    
        print("maybe LD_LIBRARY_PATH unconfigured?, import torch_scatter.scatter_cuda dies")
        pass

    # fairseq
    try:
        from transition_amr_parser.roberta_utils import extract_features_aligned_to_words_batched
    except ImportError:    
        print("fairseq installation failed")
        pass

    try:
        # transition_amr_parser
        from transition_amr_parser.amr_parser import AMRParser
    except ImportError:    
        print("transition_amr_parser installation failed")
        pass
