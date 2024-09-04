import random

def random_int(ranges = [0,100],num = 1):#生成整数
    if(ranges[0] > ranges[1]):
        print("范围错误")
        return []
    res = []
    for i in range(num):
        res.append(random.randint(ranges[0],ranges[1] + 1))
    return res
#生成浮点数
def random_float(ranges = [0,100] ,num = 1):
    if(ranges[0] > ranges[1]):
        print("范围错误")
        return []
    res = []
    for i in range(num):
        res.append(random.random()*(ranges[1] - ranges[0]) + ranges[0])
    return res

if __name__ == '__main__':
    A = random_int([2,100000],100)
    B = random_float([1,100],100)
    print(A)
    A.sort()
    print(A)
    print(B)
    B.sort()
    print(B)