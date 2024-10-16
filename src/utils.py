import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser


def test_results_post_process(root_dir, save_dir, path, config, run):

    PATH = root_dir

    def statistic(results):
        results = np.array(results)
        results = np.vstack((results, np.mean(results, axis=0)))
        mean = np.mean(results, axis=1)
        std = np.std(results, axis=1)
        results = np.hstack((results, mean.reshape(-1, 1), std.reshape(-1, 1)))
        results = np.vstack(
            (results, np.std(results[:-1, :], axis=0), np.median(results[:-1, :], axis=0)))
        return results

    def leave_subject(results_all):
        all = np.array(
            [results_all[0][:, -2], results_all[1][:, -2]]).transpose()
        all = np.hstack((all, np.mean(all, axis=1).reshape(-1, 1)))
        all[-2, -1] = np.std(all[:-3, -1])
        all[-1, -1] = np.median(all[:-3, -1])
        return all

    if config == 'leave_subject':
        results = [[] for _ in range(16)]
        for i in range(16):
            for j in range(run):
                with open(f'{PATH}/{save_dir}/{path}/version_{i*run+j}/test_accuracy.json', 'r') as f:
                    acc = json.load(f)
                    results[i].append(acc['test/accuracy'])
        results = statistic(results)
        with open(f'{PATH}/{save_dir}/{path}/test_accuracy.csv', 'w') as f:
            df = pd.DataFrame(results, index=[f'S{i}' for i in range(1, 17)] + ['mean', 'std', 'median'], columns=[f'run{i+1}' for i in range(run)] + ['mean', 'std'])
            df.to_csv(f)

    elif config == 'all_subject_leave_story':
        results_all = []
        for i in range(2):
            results = [[] for _ in range(16)]
            for j in range(run):
                with open(f'{PATH}/{save_dir}/{path}/version_{i*run+j}/test_accuracy.json', 'r') as f:
                    acc = json.load(f)
                    for k in range(16):
                        results[k].append(acc[f'S{k+1}'])
            results = statistic(results)
            with open(f'{PATH}/{save_dir}/{path}/leave_story{i+1}.csv', 'w') as f:
                df = pd.DataFrame(results, index=[f'S{m}' for m in range(
                    1, 17)] + ['mean', 'std', 'median'],  columns=[f'run{n+1}' for n in range(run)] + ['mean', 'std'])
                df.to_csv(f)
            results_all.append(results)
        all = leave_subject(results_all)
        with open(f'{PATH}/{save_dir}/{path}/test_accuracy_all.csv', 'w') as f:
            df = pd.DataFrame(all, index=[f'S{i}' for i in range(1, 17)] + ['mean', 'std', 'median'], columns=['leave_story1', 'leave_story2', 'mean'])
            df.to_csv(f)

    elif config == 'all_subject_per_trial':
        results = [[] for _ in range(16)]
        for i in range(run):
            with open(f'{PATH}/{save_dir}/{path}/version_{i}/test_accuracy.json', 'r') as f:
                acc = json.load(f)
                for j in range(16):
                    results[j].append(acc[f'S{j+1}'])
        results = statistic(results)
        with open(f'{PATH}/{save_dir}/{path}/test_accuracy.csv', 'w') as f:
            df = pd.DataFrame(results, index=[f'S{i}' for i in range(1, 17)] + ['mean', 'std', 'median'], columns=[f'run{i+1}' for i in range(run)] + ['mean', 'std'])
            df.to_csv(f)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--root_dir', type=str)
    args.add_argument('--save_dir', type=str)
    args.add_argument('--path', type=str)
    args.add_argument('--config', type=str)
    args.add_argument('--run', type=int)
    args = args.parse_args()
    test_results_post_process(args.root_dir, args.save_dir, args.path, args.config, args.run)
