import setuptools
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())
        
def requirements():
    with open('requirements.txt', "r") as fh:
        return [x for x in fh.read().split('\n') if x]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='gaitpy',
                 version='1.5.3',
                 description='Read and process raw vertical accelerometry data from a lumbar sensor during gait; calculate clinical gait characteristics.',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='http://github.com/matt002/gaitpy',
                 packages=setuptools.find_packages(),
                 author='Matthew Czech',
                 author_email='czech1117@gmail.com',
                 keywords=['gait', 'lumbar', 'sensor', 'digital', 'wearable', 'python', 'inverted pendulum', 'czech'],
                 classifiers=["Programming Language :: Python :: 3.6",
                              "License :: OSI Approved :: MIT License"],
                 license='MIT',
                 zip_safe=False,
                 cmdclass={'build_ext':build_ext},
                 setup_requires=['numpy==1.13.3'],
                 install_requires=requirements(),
                 include_package_data=True)
