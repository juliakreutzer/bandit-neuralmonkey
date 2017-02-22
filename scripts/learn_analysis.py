import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plotBLEU(logfile):
    with open(logfile, "r") as f:
        scores = []
        iterations = []
        stop = False
        for line in f:
            if "Validation" in line:
                stop = True
            elif stop == True:
                stop = False
                i = int(line.strip().split()[5])
                n = float(line.strip().split()[-1].split("\x1b")[0])
                scores.append(n)
                iterations.append(i)
    name = logfile.split("/")[1]
    if "cont" in name:
        style = "r"
    elif "bin" in name:
        style = "b"
    else:
        style = "g"
    if "x" in name:
        style += "-"
    else:
        style += "--"
    plt.plot(iterations, scores, style, label=name)
    #plt.xlim((0,40444*15))
    #plt.ylim((0.2555,0.2725))
    plt.xlabel("iterations")
    plt.ylabel("BLEU")
    #plt.legend()
    plt.title("BLEU on dev for {}".format(name))
    plotfile = "{}.BLEU.png".format(logfile)
    plt.savefig(plotfile)
    print("Saved BLEU plot in {}".format(plotfile))
    plt.close()


def loadArrays(dir):
    files = [f for f in os.listdir(dir) if ".npy" in f]
    arrays = {int(f.split(".npy")[0]): np.load("{}/{}".format(dir,f)) for f in files}
    print("Loaded {} arrays from {}".format(len(arrays), dir))
    return(arrays)


def loadData(logfile):
    graddir = logfile.replace("experiment", "gradients")
    upddir = logfile.replace("experiment", "updates")
    rewarddir = logfile.replace("experiment", "rewards")

    gradients = loadArrays(graddir)
    updates = loadArrays(upddir)
    rewards = loadArrays(rewarddir)

    return gradients, updates, rewards


def computeVariance(gradients, name):
    expected_grad = np.mean(list(gradients.values()), 0)
    max_var = 0
    sum_var = 0
    for i in sorted(gradients):
        var_t = np.linalg.norm(gradients[i]-expected_grad, 2)**2
        sum_var += var_t
        if var_t > max_var:
            max_var = var_t
    print("Max variance for {}: {}".format(name, max_var))
    mean_var = sum_var/len(gradients)
    print("Mean variance for {}: {}".format(name, mean_var))
    return max_var, mean_var


def computeAvgSqNorm(gradients, name):
    avg_norm = np.mean([np.linalg.norm(g, 2)**2 for g in gradients.values()])
    print("Mean squared norm for {}: {}".format(name, avg_norm))
    return avg_norm


def plotGradSqNorm(logfile, gradients, name):
    sorted_its = sorted(gradients)
    norms = [np.linalg.norm(gradients[i], 2)**2 for i in sorted_its]
    plt.plot(sorted_its, norms)
    s = np.std(norms)
    plt.ylim(min(norms)-2*s, max(norms)+2*s)
    plt.xlabel("iterations")
    plt.title("{} squared norms for {}".format(name[:-1], logfile.split("/")[1]))
    plotfile = "{}.{}-norm.png".format(logfile, name)
    plt.savefig(plotfile)
    print("Saved squared norm plot in {}".format(plotfile))
    plt.close()

def plotAvgRewards(logfile, rewards):
    sorted_its = sorted(rewards)
    avg_rewards = [np.mean(rewards[i])*100 for i in sorted_its]
    plt.plot(sorted_its, avg_rewards)
    plt.xlabel("iterations")
    plt.title("avg rewards for {}".format(logfile.split("/")[1]))
    plotfile = "{}.avg-rewards.png".format(logfile)
    plt.savefig(plotfile)
    print("Saved avg rewards plot in {}".format(plotfile))
    plt.close()

def plotCumRewards(logfile, rewards):
    sorted_its = sorted(rewards)
    sum_rewards = [np.sum(rewards[i]) * 100 for i in sorted_its]
    cum_rewards = [np.sum(sum_rewards[:i]) for i in range(len(sorted_its))]
    plt.plot(sorted_its, cum_rewards)
    plt.xlabel("iterations")
    plt.title("cumulative rewards for {}".format(logfile.split("/")[1]))
    plotfile = "{}.cum-rewards.png".format(logfile)
    plt.savefig(plotfile)
    print("Saved cumulative rewards plot in {}".format(plotfile))
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Analysis of learning process")
    p.add_argument("logfile")
    p.add_argument("--plotBLEU", action="store_true")
    args = p.parse_args()

    if args.plotBLEU:
        plotBLEU(args.logfile)

    gradients, updates, rewards = loadData(args.logfile)

    computeVariance(gradients, "gradients")
    computeVariance(updates, "updates")
    computeAvgSqNorm(gradients, "gradients")
    computeAvgSqNorm(updates, "updates")

    plotGradSqNorm(args.logfile, gradients, "gradients")
    plotGradSqNorm(args.logfile, updates, "updates")

    plotAvgRewards(args.logfile, rewards)
    plotCumRewards(args.logfile, rewards)


if __name__=="__main__":
    main()