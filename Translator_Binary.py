import numpy as np

alphabet = "_ abcdefghijklmnopqrstuvwxyz"
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

def indexToBinaryArray(index):
    res = np.zeros(5)
    temp = [int(x) for x in bin(index)[2:]]

    start = 5 - len(temp)
    res[start:] = temp

    return res

index_to_code = dict((i, indexToBinaryArray(i)) for i in xrange(28))

def TextToOneHotSymbolCode(data, lenght):
    ql = 200
    res_data = np.zeros((lenght, 2, ql, 5), dtype=np.int8)

    for j in range(lenght):
        res_row = res_data[j]
        q1 = data.iloc[j, 0]
        q2 = data.iloc[j, 1]

        for ii in range(ql):
            symbol = '_'
            if(ii<len(q1)):
                symbol = q1[ii]

            if(symbol in char_to_int):
                index = char_to_int[symbol]
                res_row[0, ii, :] = index_to_code[index]

            if (ii < len(q2)):
                symbol = q2[ii]

            if(symbol in char_to_int):
                index = char_to_int[symbol]
                res_row[1, ii , :] = index_to_code[index]

        # if(j%1000==0):
        #     print(j)

    return res_data