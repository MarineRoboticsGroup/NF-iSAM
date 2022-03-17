from setuptools import setup

setup(
    name='NF-iSAM',
    version='0.0',
    packages=['slam', 'flows', 'stats', 'utils', 'factors', 'sampler', 'geometry', 'manhattan_world_with_range'],
    package_dir={'': 'src'},
    url='',
    license='MIT',
    author='Qiangqiang Huang, Can Pu',
    author_email='qiangqiang.huang.me@gmail.com, pucanmail@gmail.com',
    description='Non-Gaussian solvers for SLAM problems'
)
