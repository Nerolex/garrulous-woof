# -*- coding: utf-8 -*-

def toPowerOfTen(k):
    return ("%.E" % k)


def secondsToHourMin(s):
    '''
    @param s:
    @return:
    s -> HH:MM
    '''
    _m, _s = divmod(s, 60)
    _h, _m = divmod(_m, 60)
    result = "%dh %02dm" % (_h, _m)
    return result


def secondsToMinSec(s):
    '''
    @param s:
    @return:
    s->MM:SS
    '''
    _m, _s = divmod(s, 60)
    _s = round(_s, 0)
    result = "%dm %02ds" % (_m, _s)
    return result


def secondsToSecMilsec(s):
    '''
    @param s:
    @return:
    s-> SS:MSMS
    '''
    _s = s
    _ms = round(s % 1, 3) * 1000
    result = "%ds %03dms" % (_s, _ms)
    return result


def secondsToMilsec(s):
    '''
    @param s:
    @return:
    s-> MSMS
    '''
    _s = s
    _ms = round(s, 3) * 1000
    result = "%02dms" % (_ms)
    return result


def convertToLibSvm(data):
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
