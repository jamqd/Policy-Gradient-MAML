import numpy as np
from matplotlib import pyplot as plt


def plotGraphs(env, hidden_dims, grad_steps=3, show=False):
    data_path = './performance_data/' + env + '/' + "_".join([str(n) for n in hidden_dims]) + '/'

    maml_train = np.load(data_path + "pg_train_maml_train_rewards.npy")
    maml_val = np.load(data_path + "pg_train_maml_val_rewards.npy")

    scratch_train = np.load(data_path + "pg_train_scratch_train_rewards.npy")
    scratch_val = np.load(data_path + "pg_train_scratch_val_rewards.npy")

    pretrain_train = np.load(data_path + "pg_train_pretrain_train_rewards.npy")
    pretrain_val = np.load(data_path + "pg_train_pretrain_val_rewards.npy")

    plt.figure()
    graph_ticks = min(len(maml_train), grad_steps)
    x_axis = range(graph_ticks)
    plt.plot(x_axis, maml_train[:graph_ticks], label="MAML")
    plt.plot(x_axis, scratch_train[:graph_ticks], label="Scratch")
    plt.plot(x_axis, pretrain_train[:graph_ticks], label="Pretrain")
    plt.xticks(x_axis)
    plt.legend(loc="best")
    plt.xlabel("Number of Gradient Steps")
    plt.ylabel("Rewards")
    plt.title(env + ", " +  str(hidden_dims) + ' (Validation)')
    plt.savefig("./graphs/" + env + "_" +  "_".join([str(n) for n in hidden_dims]) + ".png")
    if show:
        plt.show()

def plotGraphsHidden(env, grad_steps=3, show=False):
    data_path = './performance_data/' + env + '/' 

    maml_train_1 = np.load(data_path +  "128/pg_train_maml_train_rewards.npy")
    maml_val_1 = np.load(data_path + "128/pg_train_maml_val_rewards.npy")

    maml_train_2 = np.load(data_path +  "128_128/pg_train_maml_train_rewards.npy")
    maml_val_2 = np.load(data_path + "128_128/pg_train_maml_val_rewards.npy")

    maml_train_3 = np.load(data_path +  "128_128_128/pg_train_maml_train_rewards.npy")
    maml_val_3 = np.load(data_path + "128_128_128/pg_train_maml_val_rewards.npy")

    

    plt.figure()
    graph_ticks = min(len(maml_train_1), grad_steps)
    x_axis = range(graph_ticks)
    plt.plot(x_axis, maml_val_1[:graph_ticks], label="1 Hidden")
    plt.plot(x_axis, maml_val_2[:graph_ticks], label="2 Hidden")
    plt.plot(x_axis, maml_val_3[:graph_ticks], label="3 Hidden")
    plt.xticks(x_axis)
    plt.legend(loc="best")
    plt.xlabel("Number of Gradient Steps")
    plt.ylabel("Rewards")
    plt.title(env + ' (Validation)')
    plt.savefig("./graphs/" + env + "_" +  "hidden.png")
    if show:
        plt.show()

if __name__ == "__main__":
    # plotGraphs("HalfCheetahForwardBackward-v1", [128], show=True)
    # plotGraphs("HalfCheetahForwardBackward-v1", [128,128], show=True)
    # plotGraphs("HalfCheetahForwardBackward-v1", [128,128, 128], show=True)

    plotGraphs("HumanoidForwardBackward-v1", [128], show=True)
    plotGraphs("HumanoidForwardBackward-v1", [128,128], show=True)
    plotGraphs("HumanoidForwardBackward-v1", [128,128,128], show=True)
    plotGraphs("AntForwardBackward-v1", [128] ,show=True)
    plotGraphs("AntForwardBackward-v1", [128,128], show=True)
    plotGraphs("AntForwardBackward-v1", [128,128, 128], show=True)

