import os
def change_file(dir_path,Str = ""):
    files = os.listdir(dir_path)
    for f in files :
        oldname = os.path.join(dir_path,f)
        newname = os.path.join(dir_path,Str + f)#原名字前添加
        os.rename(oldname,newname)
        print(oldname,'--->',newname)

change_file("C:\\Users\\ASUS\\Downloads\\压缩包","new")
