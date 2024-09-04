import os
import numpy as np
from PIL import Image
import cv2
#重设大小
def resize_image(width, height, infile, outfile):
    im = Image.open(infile)
    # print(np.array(im))
    # print(im.size)
    out = im.resize((width,height),Image.Resampling.LANCZOS)
    out.save(outfile)

    return out

#灰度图转换
def rgb2gray(infile,outfile):
    im = Image.open(infile)
    L = im.convert('L')#转换为灰度图像选择参数'L'
    L.save(outfile)
    return L

#图像增强，gamma变换
def gamma_transfer(infile,outfile,power1 = 1):#power1: 伽马校正的幂指数，默认为1，即不进行校正。
    
    im = cv2.imread(infile)
    if len(im.shape) == 3:
        # 如果是彩色图像，将BGR格式转换为RGB格式
        # 因为OpenCV默认读取的图像格式是BGR，而许多图像处理操作通常使用RGB格式
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

     # 进行伽马校正
    # 将图像数据标准化到0-1范围，然后进行幂运算，再乘以255恢复到0-255范围
    im = 255 * np.power(im/255,power1)
    # 伽马校正后，可能会有像素值超出0-255的范围，需要将这些值限制在0-255之间
    im[im>255] = 255
     # 将图像数据类型转换为uint8，因为OpenCV处理图像时通常使用这种数据类型
    out = im.astype(np.uint8)
    cv2.imwrite(outfile,out)
    return im

def Contrast_and_Brightness(infile,outfile,alpha,beta):
    """使用公式f(x) = d.g(x) + b"""
    #a调节对比度，b调节亮度
    # alpha: 对比度调整因子，范围为0-1，0表示无对比度调整，1表示原始对比度。
    # beta: 亮度调整值，可以为负数或正数，负数表示调暗，正数表示调亮。
    im = cv2.imread(infile)
    # 创建一个全零的图像，用于计算对比度和亮度调整后的图像
    blank = np.zeros(im.shape,im.dtype)
     # 使用cv2.addWeighted函数计算对比度和亮度调整后的图像
    # 该函数的参数分别是：
    # 1. 原图像
    # 2. 对比度调整因子（alpha）
    # 3. 亮度调整值（beta）
    # 4. 亮度调整的补偿值（1-alpha），用于避免对比度过大或过小导致亮度变化不均匀
    # 5. 调整后的图像
    dst = cv2.addWeighted(im,alpha,blank,1-alpha,beta)

    cv2.imwrite(outfile,dst)
    return dst 

file1 = "D:\\learn-git\\gitee_repo\\鲜花\\picture"
file2 = 'D:\\learn-git\\gitee_repo\\鲜花\\resize'
# files1 = os.listdir("D:\\learn-git\\gitee_repo\\鲜花\\picture")
# files2 = os.listdir('D:\\learn-git\\gitee_repo\\鲜花\\resize')
for x in os.listdir(file1):
    infile = os.path.join(file1,x)
    outfile = os.path.join(file2,x)
    re_image = resize_image(100,100,infile,outfile)
# rgb2gray(outfile,outfile)
