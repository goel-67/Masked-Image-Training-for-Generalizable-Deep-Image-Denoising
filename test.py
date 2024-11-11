import torch

# Check if CUDA is available
print("Is CUDA available?:", torch.cuda.is_available())

# Check the version of CUDA PyTorch is using
print("CUDA version PyTorch is built with:", torch.version.cuda)

# Get the current CUDA device being used
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
