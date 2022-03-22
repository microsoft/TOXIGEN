from setuptools import setup

setup(
    name='toxigen',
    version='0.1.0',    
    description='Code support for ToxiGen',
    url='https://github.com/microsoft/toxigen',
    author='Thomas Hartvigsen',
    author_email='tomh@mit.edu',
    license='BSD 2-clause',
    packages=['toxigen'],
    install_requires=['torch',                     
                      'numpy',
                      #'tokenizers==0.6.0',
                      'transformers==4.16.2',
                      'pandas',                     
                      ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python :: 3.8',
    ],
)
