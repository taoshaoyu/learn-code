# 对象 装饰 函数

class Decorator:             # 创建作为装饰器类对象的类
     def __init__(self, arg1, arg2):
         print('执行类Decorator的__init__()方法')
         self.arg1 = arg1
         self.arg2 = arg2
         
     def __call__(self, f):     # 参数 f，是
         print('执行类Decorator的__call__()方法')
         def wrap(*args):
             print('执行wrap()')
             print('装饰器参数：', self.arg1, self.arg2)
             print('执行' + f.__name__ + '()')
             f(*args)
             print(f.__name__ + '()执行完毕')
         return wrap            # 返回 wrap 函数，也就是被装饰过后的函数
     
@Decorator('Hello', 'World')    # 实例化装饰器类，并用来装饰函数
def example(a1, a2, a3):
     print('传入example()的参数：', a1, a2, a3)
     
print('装饰完毕')

print('准备调用example()')
example('Wish', 'Happy', 'EveryDay')
print('测试代码执行完毕')