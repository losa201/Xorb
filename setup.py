from setuptools import setup, find_packages

setup(
    name='xorbfw',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'networkx',
        'gym',
    ],
    entry_points={
        'console_scripts': [
            'xorbfw-demo = xorbfw.demonstration:main',
            'xorbfw-validate = xorbfw.validation:main',
        ],
    },
    author='XORB Team',
    author_email='xorbfw@example.com',
    description='XORB Autonomous War-Gaming Framework',
    long_description='Advanced simulation framework for adversarial scenario training and analysis',
    url='https://github.com/xorbfw/xorbfw',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
