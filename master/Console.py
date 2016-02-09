import datetime


def write(str):
    debug = open("debug.txt", mode='a')
    time_str = "[" + datetime.datetime.now().strftime('%H:%M:%S') + "]: "
    print(time_str + str + "\n")
    debug.write(time_str + str + "\n")
