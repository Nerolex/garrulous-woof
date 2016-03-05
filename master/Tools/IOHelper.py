# -*- coding: utf-8 -*-

import datetime

'''
Helper class for input and output.
Reports strings to the console and writes external files.
'''

def write(str):
    debug = open("debug.txt", mode='a')
    time_str = "[" + datetime.datetime.now().strftime('%H:%M:%S') + "]: "
    print(time_str + str + "\n")
    debug.write(time_str + str + "\n")


def createLongFile(data, date, end_result):
    filestring = data + "_long_" + date
    filestring = filestring.replace(" ", "_")
    filestring = filestring.replace(":", "_")
    try:
        file = 'master/output/' + filestring + ".csv"
        output = open(file, 'a')
    except(Exception):
        file = 'output/' + filestring + ".csv"
        output = open(file, 'a')

    header = ["k, Points gaussian,", "Points linear,", "C linear,", "C gauss,", "gamma gauss,", "number SVs gauss,",
              "time to fit,", "Gauss,", "Linear,", "Overhead,", "time to predict,", "Error\n"]
    for col in header:
        output.write(col)
    output.write("\n")

    for i in range(len(end_result.n_lin)):
        output.write(
            str(end_result.k[i]) + "," +
            str(end_result.n_gauss[i]) + "," +
            str(end_result.n_lin[i]) + "," +
            str(end_result.c_lin[i]) + "," +
            str(end_result.c_gauss[i]) + "," +
            str(end_result.gamma[i]) + "," +
            str(end_result.sv_gauss[i]) + "," +
            str(end_result.time_fit[i]) + "," +
            str(end_result.time_fit_gauss[i]) + "," +
            str(end_result.time_fit_linear[i]) + "," +
            str(end_result.time_fit_overhead[i]) + "," +
            str(end_result.time_predict[i]) + "," +
            str(end_result.error[i]) + "\n"
        )


def createShortFile(data, date, end_result):
    filestring = data + "_short_" + date
    filestring = filestring.replace(" ", "_")
    filestring = filestring.replace(":", "_")
    try:
        file = 'master/output/' + filestring + ".csv"
        output = open(file, 'a')
    except(Exception):
        file = 'output/' + filestring + ".csv"
        output = open(file, 'a')

    header = ["k,", "Error,", "Time Fit,", "Time Predict"]
    for col in header:
        output.write(col)
    output.write("\n")
    for i in range(len(end_result.n_lin)):
        output.write(
            str(end_result.k[i]) + "," +
            str(end_result.error[i]) + "," +
            str(end_result.time_fit[i]) + "," +
            str(end_result.time_predict[i]) + "\n"
        )
