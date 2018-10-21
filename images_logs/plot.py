import matplotlib.pyplot as plt

def plot_fig(log_path, save_dir):
    itr = []
    acc = []

    for line in open(log_path):
        x = line.split(' ')
        itr.append(float(x[1]))
        acc.append(float(x[len(x)-1]))

    plt.plot(itr, acc)
    plt.title('GradientDescent')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.savefig(save_dir)

def main():
    plot_fig('training.log', 'grad_desc.png')

if __name__ == '__main__':
    main()

