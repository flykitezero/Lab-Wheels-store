# 提供一些常用的CV相关python代码
用于分享大家常用的轮子，增加效率，欢迎大家将常用的功能添加进该项目中！
该项目作为公用库，可以任意取用，添加时请注明提交人，详尽写好注释，方便使用，也欢迎各位自由发挥。
全部代码存放于Main Store，进入后在网页中**ctrl+F**即可，或者pull到本地查找。

简单格式如下：
```ruby
# -------------------   
提交人：  
函数简介：  
提交日期：  
特殊导入包：
# ------------------------ 
（函数名）  
'''  
para1：  
para2：  
...  

return：  
'''  
（以下为代码主体）  
```



e.g:  
```ruby
# -------------------    
提交人：SZ  
功能简介：读取文件中的3通道图片，转为灰度图  
特殊导入包：os,cv2
提交日期：2021/7/21  
# ------------------------ 

def JPG2GRAY(source_Dir, new_Dir):  
    '''  
    :param source_Dir: 要处理的文件夹路径  
    :param new_Dir: 保存的路径  
    :return:无  


    # 判断是否存在保存路径，不存在则新建
    if not os.path.exists(new_Dir):
        os.makedirs(new_Dir)
    # 处理图片
    for filename in os.listdir(source_Dir):
        # 打印文件名
        print(filename)
        # 读取图片
        img = cv2.imread(os.path.join(source_Dir, filename))
        # 将图片转换为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 保存图片
        cv2.imwrite(os.path.join(new_Dir, filename), img)

        print('处理成功')
```
