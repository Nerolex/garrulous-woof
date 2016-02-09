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
