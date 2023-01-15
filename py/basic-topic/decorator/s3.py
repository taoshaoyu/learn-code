# 函数装饰函数，装饰器带参数
from functools import wraps

def double_d( pa=1, pb=2):
    def actual_func(func):
        @wraps(func)
        def inner_func(*args, **kwargs):
            print("pa=%d, pb=%d"%(pa,pb))
            return func(*args, **kwargs)
        return inner_func
    return actual_func

@double_d(1,2)
def foo():
    print("this is foo")

foo()



