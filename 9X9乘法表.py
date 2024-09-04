
def multi():
    s = ''
    for i in range(1,10):
        for j in range(1,i + 1):
            s += '{} * {} = {}'.format(i,j,i * j) + ' '
        s += '\n'
    return s

print(multi())