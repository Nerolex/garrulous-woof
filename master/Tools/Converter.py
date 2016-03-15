# -*- coding: utf-8 -*-

from ParameterTuning import Gridsearcher

'''
This is a helper class for string conversions.
'''

def convertToLibSvm(data):
    '''
    Specific method for converting UCI data into libsvm format.

    :param data: String, name of the data. Search is done automatically in the data directory.
    :return: None. Converted file is save to a new file.
    '''
    filestring = "../data/" + data + "/" + data
    filetypes = ['.txt', '.t']

    filestring1 = filestring + filetypes[0]
    filestring2 = filestring + filetypes[1]

    convertFile(filestring1)
    convertFile(filestring2)


def convertFile(filestring):
    file1 = open(filestring, 'r')
    toWrite = []
    for line in file1:
        line = line.strip("\n")
        line = line.split(" ")
        new_line = []
        label = line[len(line) - 1]
        new_line.append(label + " ")
        for i in range(len(line) - 2):
            new_line.append(str(i + 1) + ":" + line[i].strip(" ") + " ")
        new_line.append("\n")
        toWrite.append(new_line)
    file1_output = open(filestring + ".conv", "w")
    for row in toWrite:
        for col in row:
            file1_output.write(col)


def convertParamsToCsv(data, filestring):
    cLin, cGauss, gamma = Gridsearcher.loadParametersFromFile(data)
    k = [0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0]
    output = open(filestring, 'w')
    output.write("k, C Linear, C Gauss, Gamma Gauss\n")
    output.write(str(k[0]) + "," + str(cLin) + ",,\n")
    for i in range(1, 9, 1):
        output.write(str(k[i]) + "," + "," + str(cGauss[i + 1]) + "," + str(gamma[i + 1]) + "\n")


def convertParamsToCorrect(data, filestring):
    cLin, cGauss, gamma = Gridsearcher.loadParametersFromFile(data)
    output = open(filestring, 'w')
    output.write(str(cLin) + "\n")

    c_gauss = ""
    cGauss.pop()
    cGauss.insert(0, 0.0)
    for element in cGauss:
        c_gauss += str(element) + ","
    c_gauss = c_gauss.rstrip(",")
    c_gauss += "\n"
    output.write(c_gauss)

    g_amma = ""
    gamma.pop()
    gamma.insert(0, 0.0)
    for element in gamma:
        g_amma += str(element) + ","
    g_amma = g_amma.rstrip(",")
    g_amma += "\n"
    output.write(g_amma)
