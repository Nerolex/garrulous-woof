import warnings


def loadParametersFromFile(data):
    filestring = "output/" + data + "-params.txt"
    file_ = open(filestring, 'r')

    i = 0
    c_lin = 0
    c_gauss = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    gamma = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for line in file_:
        # 0  C_lin
        # 1  C_Gauss
        # 2  gamma
        if i == 0:
            line = line.strip("\n")
            c_lin = float(line)
        if i == 1:
            values = line.split(",")
            for j in range(len(values) - 1):
                c_gauss[j] = values[j]
                if values[j] == 0:
                    c_gauss[j] == c_gauss[j - 1]
        if i == 2:
            values = line.split(",")
            for j in range(len(values) - 1):
                gamma[j] = values[j]
                if values[j] == 0:
                    gamma[j] == gamma[j - 1]
        i += 1
    return c_lin, c_gauss, gamma

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print loadParametersFromFile("ijcnn")
