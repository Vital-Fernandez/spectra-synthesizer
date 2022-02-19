import numpy as np

VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]


def int_to_roman(num):
    i, roman_num = 0, ''
    while num > 0:
        for _ in range(num // VAL_LIST[i]):
            roman_num += SYB_LIST[i]
            num -= VAL_LIST[i]
        i += 1
    return roman_num


def ftau_func(tau, temp, den, a, b, c, d):
    return 1 + tau / 2.0 * (a + (b + c * den + d * den * den) * temp / 10000.0)


def assignFluxEq2Label(labelsList, ionsList, recombLabels=['H1', 'He1', 'He2']):
    eqLabelArray = np.copy(ionsList)

    for i in range(eqLabelArray.size):
        if eqLabelArray[i] not in recombLabels:
            # TODO integrate a dictionary to add corrections
            if labelsList[i] != 'O2_7319A_b':
                eqLabelArray[i] = 'metals'
            else:
                eqLabelArray[i] = 'O2_7319A_b'

    return eqLabelArray


