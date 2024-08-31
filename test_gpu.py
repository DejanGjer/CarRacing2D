# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. GPU will be used.")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Perform a simple tensor operation
try:
    # Create two tensors on the GPU
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)

    # Perform a matrix multiplication
    c = torch.matmul(a, b)

    print("Result of the matrix multiplication:")
    print(c)
except Exception as e:
    print(f"An error occurred: {e}")

