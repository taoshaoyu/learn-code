# setup.py
from distutils.core import setup, Extension
import sysconfig

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-std=c++11", "-Wall", "-Wextra"]
extra_compile_args += ["-g","-O0"]


module1 = Extension('demo',
                    sources = ['hello.c'],
                    extra_compile_args=extra_compile_args)

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])