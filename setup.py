import os
import os.path as osp
import glob
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
# getenv从系统中获得环境变量，第一个为获取的变量名，第二个参数为环境变量的默认值
if os.getenv('FORCE_CUDA', '0') == '1':
    WITH_CUDA = True
if os.getenv('FORCE_CPU', '0') == '1':
    WITH_CUDA = False

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


def get_extensions():
    Extension = CppExtension
    define_macros = []
    # extra_compile_args = {'cxx': []}

    if WITH_CUDA:
        Extension = CUDAExtension
        # define_macros += [('WITH_CUDA', None)]
        # nvcc_flags = os.getenv('NVCC_FLAGS', '')
        # nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        # nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr']
        # extra_compile_args['nvcc'] = nvcc_flags

    #  __file__为当前文件路径, dirname去掉文件名返回路径, join用来连接路径
    extensions_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'csrc')
    # 匹配所有的符合条件的文件，并将其文件名以list的形式返回
    main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))
    extensions = []
    for main in main_files:
        # os.sep在Windows上，文件的路径分隔符是'\'，os.sep在Linux上是'/'
        name = main.split(os.sep)[-1][:-4]

        sources = [main]

        path = osp.join(extensions_dir, f'{name}_cpu.cpp')
        # os.path.exists判断括号里的文件是否存在的意思，括号内的可以是文件路径。
        if osp.exists(path):
            sources += [path]

        path = osp.join(extensions_dir, f'{name}_cuda.cu')
        if WITH_CUDA and osp.exists(path):
            sources += [path]

        extension = Extension(
            'my_scatter._' + name,
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            # extra_compile_args=extra_compile_args,
        )
        extensions += [extension]

    return extensions


install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']
test = get_extensions()
pass
# setup(
#     name='my_scatter',
#     python_requires='>=3.6',
#     install_requires=install_requires,
#     ext_modules=get_extensions(),
#     cmdclass={
#         'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
#     },
#     packages=find_packages(),
# )
