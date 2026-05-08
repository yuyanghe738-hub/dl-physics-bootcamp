import platform

import matplotlib
import numpy as np
import torch


print("Python:", platform.python_version())
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("Device suggestion:", "mps" if torch.backends.mps.is_available() else "cpu")

