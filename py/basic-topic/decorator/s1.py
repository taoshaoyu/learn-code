# 函数 装饰 函数
def hello(fn):
    def wrapper1():
        print("hello, %s" % fn.__name__)
        fn()
        print("goodby, %s" % fn.__name__)
    return wrapper1

@hello
def foo():
    print("i am foo")

foo()

a=hello(foo)

a()