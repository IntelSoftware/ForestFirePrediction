import torch
import intel_extension_for_pytorch

import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from config import device


def estimate_batch_size(memory_usage_data, total_memory):
    """estimates the maximum feasible batch size using a polynomial regression model.

    params:
    memory_usage_data (list of tuple): List containing data points of the form (batch_size, memory_allocated)
    total_memory (int): Total GPU memory available

    return:
    int: Estimated maximum feasible batch size
    """
    if not memory_usage_data:
        print("No data to process.")
        return
    X, y = zip(*memory_usage_data)
    X = np.array(X).reshape(-1, 1)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)
    feasible_batch_sizes = np.arange(1, np.max(X))[
        model.predict(np.arange(1, np.max(X)).reshape(-1, 1)) < total_memory
    ]
    return np.max(feasible_batch_sizes) if len(feasible_batch_sizes) > 0 else 1


def test_batch_size(model, batch_size, input_size, device):
    """For a given batch size, test the model by performing a forward and backward pass.

    params:
    model (torch.nn.Module): The model to be tested
    batch_size (int): The batch size to be tested
    input_size (tuple): The dimensions of the model input
    device (str): The device on which to run the test

    return:
    float: The amount of GPU memory allocated
    """
    data = torch.rand(batch_size, *input_size, device=device)
    targets = torch.randint(0, 2, (batch_size,), device=device)
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, targets)
    loss.backward()
    return torch.xpu.memory_allocated()


def find_optimal_batch_size(
    model, input_size, min_batch_size=32, max_batch_size=512, num_samples=10
):
    """find optimal batch size that maximizes memory usage without going out of memory.

    params:
    model (torch.nn.Module): The model to be tested
    input_size (tuple): The dimensions of the model input
    min_batch_size (int, optional): The minimum batch size to test
    max_batch_size (int, optional): The maximum batch size to test
    num_samples (int, optional): The number of different batch sizes to test

    return:
    list of tuple: List containing data points of the form (batch_size, memory_allocated)
    """
    device = f"xpu:{torch.xpu.current_device()}"
    model = model.to(device)
    memory_usage_data = []
    for i in tqdm(range(num_samples), desc="Finding optimal batch size"):
        mid_batch_size = (max_batch_size + min_batch_size) // 2
        try:
            memory_allocated = test_batch_size(
                model, mid_batch_size, input_size, device
            )
            memory_usage_data.append((mid_batch_size, memory_allocated))
            min_batch_size = mid_batch_size + 1
        except ImportError:
            print(f"Failed at batch size: {mid_batch_size}")
            max_batch_size = mid_batch_size - 1
    return memory_usage_data


def optimum_batch_size(model, input_size):
    """Determines the optimum batch size for the model.

    params:
    model (torch.nn.Module): The model to be tested
    input_size (tuple): The dimensions of the model input

    return:
    int: The estimated optimum batch size
    """

    if not torch.xpu.is_available():
        print(f"No XPU available., using : {torch.device('cpu')}")
        return 64
    return 64
    # todo fix this code, there is an issue with batch size finder failing
    #total_memory = torch.xpu.get_device_properties(device).total_memory
    #memory_usage_data = find_optimal_batch_size(model, input_size)
    #return estimate_batch_size(memory_usage_data, total_memory)
