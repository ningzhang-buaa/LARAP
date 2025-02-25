import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns

#sns.set_theme()

EXP_PATH = '../data/lift'
PLOT_FOLDER = '.'

MAX_EPOCH = 500

# smoothing values
def smooth(values, weight):
    smoothed = np.array(values)
    for i in range(1, smoothed.shape[0]):
        smoothed[i] = smoothed[i-1] * weight + (1 - weight) * smoothed[i]
    return smoothed

def plot_experiment(configs):

    #plt.figure(figsize=(4, 3))

    handles = []
    n=0
    for exp_config in configs["prefixes"]:

        all_runs = os.listdir(EXP_PATH)
        print(all_runs)

        runs = [x for x in all_runs if x.startswith(exp_config)]

        x = np.array(range(MAX_EPOCH))

        returns_avg = []
        returns_std = []
        print(runs)
        n=n+1

        for r in runs:

            folder = os.listdir(os.path.join(EXP_PATH, r))
            print(folder[0])
            print(len(folder))
            assert len(folder) == 1

            df = pd.read_csv(os.path.join(EXP_PATH, r, folder[0], 'progress.csv'))
            if n<3:
                r_mean = df['eval/Returns Mean'].to_numpy()
                r_std = df['eval/Returns Std'].to_numpy()
            else:
                r_mean = df['evaluation/Returns Mean'].to_numpy()
                r_std = df['evaluation/Returns Std'].to_numpy()
            #print(r_mean.shape)

            assert r_mean.shape[0] >= MAX_EPOCH, r_mean.shape[0]
            assert r_std.shape[0] >= MAX_EPOCH

            r_mean = r_mean[:MAX_EPOCH]
            r_std = r_std[:MAX_EPOCH]

            returns_avg.append(r_mean)
            returns_std.append(r_std)

        weight = 0.99 # smoothing coefficient
        for idx, (means, stds) in enumerate(zip(returns_avg, returns_std)):
            returns_avg[idx] = smooth(means, weight)
            returns_std[idx] = smooth(stds, weight)

        y_range = (0, 500) # reward scale

        returns_avg = np.array(returns_avg)
        returns_std = np.array(returns_std)

        y_mean = returns_avg.mean(0)
        y_std  = returns_std.mean(0)

        h, = plt.plot(x, y_mean)
        handles.append(h)

        confidence_min = y_mean - y_std
        confidence_max = y_mean + y_std

        plt.fill_between(x, confidence_min, confidence_max, alpha=0.15)

    plt.xlabel('training epochs')
    plt.ylabel('episode return')

    plt.xlim(0, MAX_EPOCH)
    #plt.ylim(0, 500)

    plt.title(configs["task_name"])

    # plt.legend(
    #     handles,
    #     ["Panda (OSC)", "Sawyer (OSC)", "Panda (Joint Velocity)", "Sawyer (Joint Velocity)"],
    #     loc=(1.05, 0.56)
    # )

    plt.legend(
        handles,
        ["maple", "maple-llm", "sac"],
        #loc=(1.05, 0.56)
    )

    #plt.tight_layout()
    #plt.show()
    file_name = os.path.join(PLOT_FOLDER, '%s.png' % configs["file_name"])
    plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0)

    print('Saving to %s' % file_name)

all_configs = [
    {
        "task_name": "Block Lifting",
        "file_name": "block_lifting",
        "prefixes": ["maple-", "llm-", "Lift-Panda"],
    }
]

for configs in all_configs:
    plot_experiment(configs)