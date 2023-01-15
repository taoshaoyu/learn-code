# 函数 装饰 类

import time

def sort_by_create_time(cls):       # 输入参数是 cls
    orig_init=cls.__init__
    def new_init(self, *args, **kwargs):    # 修改 cls 的 __init__ 部分
        orig_init(self, *args, **kwargs)
        self._created=time.time()
    cls.__init__=new_init
    return cls                      # 输出 cls

@sort_by_create_time                # 装饰类的装饰器
class ca(object):
    def __init__(self, a):
        self.a=a
    def printme(self):
        print(self._created)

ca_obj1=ca(1)
ca_obj1.printme()



