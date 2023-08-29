import torch
from torch_geometric.data import Data
import tempfile
import time
import multiprocessing as mp
import datamol as dm
import pandas as pd
from functools import partial
import os


# Create a function to generate a random pyg graph
def generate_graph():
    x = torch.randn(25, 16)
    edge_index = torch.randint(0, 25, (2, 50))
    data = Data(x=x, edge_index=edge_index)
    return data


def save_data(args):
    data, filename = args
    torch.save(data, filename)


def save_data_batch(args):
    for this_args in args:
        save_data(this_args)
    return [None] * len(args)


def load_data(filename):
    torch.load(filename)


def load_data_batch(filenames):
    for filename in filenames:
        load_data(filename)
    return [None] * len(filenames)


def make_input_list(data_list, tmpdirname):
    input_list = []
    for i, data in enumerate(data_list):
        folder_idx = i // 1000
        folder_dir = f"{tmpdirname}/folder_{folder_idx}"
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        filename = f"{folder_dir}/data_list_{i}.pt"
        input_list.append((data, filename))
    name_list = [input_list[ii][1] for ii in range(len(input_list))]
    return input_list, name_list


def bench_single_thread(data_list, num_processes):
    # Make a temporary directory, and save the data_list in it, and track the time
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Single thread benchmark
        input_list, name_list = make_input_list(data_list, tmpdirname)
        start_time = time.time()
        for i in range(len(data_list)):
            save_data((data_list[i], name_list[i]))
        save_elapsed_time = time.time() - start_time
        print(
            f"single SAVE --- num_files={len(data_list)} --- num_processes={num_processes} --- {save_elapsed_time} seconds ---"
        )

        # Load the data_list from the temporary directory, and track the time
        start_time = time.time()
        for i in range(len(data_list)):
            load_data(name_list[i])
        load_elapsed_time = time.time() - start_time
        print(
            f"single LOAD --- num_files={len(data_list)} --- num_processes={num_processes} --- {load_elapsed_time} seconds ---"
        )

        return save_elapsed_time, load_elapsed_time


def bench_multiprocessing(data_list, num_processes, context):
    # Using multiprocessing spawn method
    ctx = mp.get_context(context)
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_list, name_list = make_input_list(data_list, tmpdirname)
        start_time = time.time()
        with ctx.Pool(processes=num_processes) as pool:
            pool.map(save_data, input_list)
        save_elapsed_time = time.time() - start_time
        print(
            f"{context} SAVE --- num_files={len(data_list)} --- num_processes={num_processes} --- {save_elapsed_time} seconds ---"
        )

        # Load the data_list from the temporary directory, and track the time
        start_time = time.time()
        with ctx.Pool(processes=num_processes) as pool:
            pool.map(load_data, name_list)
        load_elapsed_time = time.time() - start_time
        print(
            f"{context} LOAD --- num_files={len(data_list)} --- num_processes={num_processes} --- {load_elapsed_time} seconds ---"
        )

    return save_elapsed_time, load_elapsed_time


def bench_datamol_parallelized(data_list, num_processes, batch_size):
    # Using datamol parallelize method
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_list, name_list = make_input_list(data_list, tmpdirname)
        start_time = time.time()
        dm.parallelized_with_batches(save_data_batch, input_list, batch_size, n_jobs=num_processes)
        save_elapsed_time = time.time() - start_time
        print(
            f"datamol SAVE --- num_files={len(data_list)} --- num_processes={num_processes} --- {save_elapsed_time} seconds ---"
        )

        # Load the data_list from the temporary directory, and track the time
        start_time = time.time()
        dm.parallelized_with_batches(load_data_batch, name_list, batch_size, n_jobs=num_processes)
        load_elapsed_time = time.time() - start_time
        print(
            f"datamol batch={batch_size} LOAD --- num_files={len(data_list)} --- num_processes={num_processes} --- {load_elapsed_time} seconds ---"
        )

    return save_elapsed_time, load_elapsed_time


def bench_threading(data_list, num_processes):
    # Using threading
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_list, name_list = make_input_list(data_list, tmpdirname)
        start_time = time.time()
        with mp.pool.ThreadPool(processes=num_processes) as pool:
            pool.map(save_data, input_list)
        save_elapsed_time = time.time() - start_time
        print(
            f"threading SAVE --- num_files={len(data_list)} --- num_processes={num_processes} --- {save_elapsed_time} seconds ---"
        )

        # Load the data_list from the temporary directory, and track the time
        start_time = time.time()
        with mp.pool.ThreadPool(processes=num_processes) as pool:
            pool.map(load_data, name_list)
        load_elapsed_time = time.time() - start_time
        print(
            f"threading LOAD --- num_files={len(data_list)} --- num_processes={num_processes} --- {load_elapsed_time} seconds ---"
        )

    return save_elapsed_time, load_elapsed_time


def bench_saving_loading(num_files_list, num_processes_list, save_path):
    # Check if "save_path" exists, if not, create a new file
    if not os.path.exists(save_path):
        results = {
            "method": [],
            "num_files": [],
            "num_processes": [],
            "save_elapsed_time": [],
            "load_elapsed_time": [],
        }
    else:
        results = pd.read_csv(save_path, index_col=0).to_dict(orient="list")

    methods = {
        # 'single': bench_single_thread,
        "multiprocessing_spawn": partial(bench_multiprocessing, context="spawn"),
        "multiprocessing_fork": partial(bench_multiprocessing, context="fork"),
        "multiprocessing_forkserver": partial(bench_multiprocessing, context="forkserver"),
        # 'datamol_parallelized_1': partial(bench_datamol_parallelized, batch_size=1),
        # 'datamol_parallelized_4': partial(bench_datamol_parallelized, batch_size=4),
        # 'datamol_parallelized_16': partial(bench_datamol_parallelized, batch_size=16),
        # "threading": bench_threading,
    }

    count = 0
    # Double for loop to test different number of files and processes
    for num_files in num_files_list:
        for num_processes in num_processes_list:
            for method_name, method_fn in methods.items():
                # Generate a list of random graphs
                data_list = [generate_graph() for _ in range(num_files)]

                # Run the benchmarking
                try:
                    save_elapsed_time, load_elapsed_time = method_fn(data_list, num_processes)
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

                # Add a new line to the dataframe
                # results['index'].append(results['index'][-1] + 1 if len(results['index']) > 0 else 0)
                results["method"].append(method_name)
                results["load_elapsed_time"].append(load_elapsed_time)
                results["save_elapsed_time"].append(save_elapsed_time)
                results["num_files"].append(num_files)
                results["num_processes"].append(num_processes)

                # Save the results
                results_df = pd.DataFrame(results)
                results_df.to_csv(save_path)


# main
if __name__ == "__main__":
    # bench_saving_loading(num_files_list=[32, 100, 320, 1000, 3200, 10000, 32000, 100000, 320000], num_processes_list=[1, 2, 4, 8, 16, 32], save_path='benchmark_save_load.csv')
    bench_saving_loading(
        num_files_list=[100000], num_processes_list=[1, 2, 4, 8, 16, 32], save_path="benchmark_save_load.csv"
    )
