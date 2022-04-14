import torch
import subprocess
from ipdb import set_trace


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

    if (
        (bare_metal_major != torch_binary_major)
        or (bare_metal_minor != torch_binary_minor)
    ):
        print(
            "Pytorch binaries were compiled with Cuda {} but binary {} is {}"
            .format(
                torch.version.cuda,
                cuda_dir + "/bin/nvcc",
                output[release_idx]
            )
        )


def main():

    # Pytorch and CUDA
    passed = True
    print()
    import torch
    print(f'pytorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'cuda {torch.version.cuda}')
        # happens when CUDA missconfigured
        assert torch.cuda.device_count(), "0 GPUs found"
        try:
            import apex
            print("Apex installed")
        except ImportError:
            print("Apex not installed")
        check_cuda_torch_binary_vs_bare_metal()
        if torch.cuda.get_device_capability(0)[0] < 7:
            print("GPU wont support --fp")

        # sanity check try to use CUDA
        import torch
        torch.zeros((100, 100)).cuda()

    else:
        print("\033[93mNo CUDA available\033[0m")

    try:
        import smatch
        print("smatch installed")
    except ImportError as e:
        print("\033[93msmatch not installed\033[0m")
        sucess = False

    try:
        import torch_scatter
        print("pytorch-scatter installed")
    except ImportError:
        print("\033[93mpytorch-scatter not installed\033[0m")
        passed = False

    if torch.cuda.is_available():
        try:
            import torch_scatter.scatter_cuda
            print("torch_scatter.scatter_cuda works")
        except ImportError:
            print(
                "\033[93mmaybe LD_LIBRARY_PATH unconfigured?, "
                "import torch_scatter.scatter_cuda dies\033[0m"
            )
            passed = False

    # fairseq
    try:
        import fairseq
        print("fairseq works")
    except ImportError:
        print("\033[93mfairseq installation failed\033[0m")
        passed = False

    # If we get here we passed
    if passed:
        print(f'[\033[92mOK\033[0m] correctly installed\n')
    else:
        print(f'[\033[91mFAILED\033[0m] some modules missing\n')


if __name__ == '__main__':
    main()
