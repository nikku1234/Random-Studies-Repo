from setuptools import setup, find_packages


setup(name='unetelan',
      version='0.1',
      description='Unet for IwM',
      packages=find_packages(), install_requires=['torchvision', 'numpy']
      )
