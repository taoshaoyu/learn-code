# 函数装饰函数，场景1

registry=[]
def register(fn):
    registry.append(fn)
    return fn

@register
def foo1():
    print("i am foo1")

@register
def foo2():
    print("i am foo2")

for f in registry:
    f()