from setuptools import setup

setup(
    name = 'andif',
    version = '0.1.0',
    packages = ['andif'],
    install_requires=[ 'plumbum',],
    entry_points = {
        'console_scripts': [
            'andif = andif.__main__:AnomalousDiffusion',
        ]
    })
