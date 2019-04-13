import matplotlib.pyplot as plt

def export_img(xlabel, ylabel, filename):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)

