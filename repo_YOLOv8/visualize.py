from matplotlib import pyplot as plt


def visualize_loss():
    with open ("./weights/loss.txt", "r") as f:
        lines = f.readlines()

    with open ("./weights/loss_xlarge.txt", "r") as f2:
        lines_2 = f2.readlines()

    loss = []
    iteration = []
    loss_x = []
    iteration_x = []

    for i in range(len(lines)):
        iteration.append(int(lines[i].split(',')[0].split(" ")[-1]))
        loss.append(float(lines[i].split(',')[1].split(" ")[-1]))

    for j in range(len(lines_2)):
        #if i % 10 == 0:
        iteration_x.append(int(lines_2[j].split(',')[0].split(" ")[-1]))
        loss_x.append(float(lines_2[j].split(',')[1].split(" ")[-1]))

    plt.plot(iteration, loss, label = "Nano")
    plt.plot(iteration_x, loss_x, label = "Xlarge")
    plt.legend()
    plt.xlim(0, 175000)
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.show()


def visualize_precision():
    with open("./weights/traning progress.txt", "r") as f:
        lines = f.readlines()

    with open ("./weights/precision_recall_map.txt", "r") as f2:
        lines_2 = f2.readlines()

    precision = []
    precision_x = []

    for i in range(len(lines)):
        if i + 1 <= len(lines):
            if "precision" in lines[i]:
                precision.append(float(lines[i+1][5: 10]))

    for i in range(len(lines_2)):
        precision_x.append(float(lines_2[i].split(',')[0].split(' ')[-1]))

    plt.plot(precision, label="Nano")
    plt.plot(precision_x, label="Xlarge")
    plt.legend()
    plt.xlim(0, 25)
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.show()


def visualize_map():
    with open("./weights/traning progress.txt", "r") as f:
        lines = f.readlines()

    with open ("./weights/precision_recall_map.txt", "r") as f2:
        lines_2 = f2.readlines()

    map = []
    map_x = []

    for i in range(len(lines)):
        if i + 1 <= len(lines):
            if "precision" in lines[i]:
                map.append(float(lines[i+1].split(' ')[-1]))

    for i in range(len(lines_2)):
        map_x.append(float(lines_2[i].split(',')[-1].split(' ')[-1]))

    plt.plot(map, label="Nano")
    plt.plot(map_x, label="Xlarge")
    plt.legend()
    plt.xlim(0, 25)
    plt.ylabel('mAP')
    plt.xlabel('epoch')
    plt.show()


if __name__ == "__main__":
    visualize_map()