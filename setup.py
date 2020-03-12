import setuptools

def requirements():
    with open('requirements.txt', "r") as fh:
        return [x for x in fh.read().split('\n') if x]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='gaitpy',
                 version='1.6.1',
                 description='Read and process raw vertical accelerometry data from a lumbar sensor during gait; calculate clinical gait characteristics.',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='http://github.com/matt002/gaitpy',
                 packages=setuptools.find_packages(),
                 author='Matthew Czech',
                 author_email='czech1117@gmail.com',
                 keywords=['gait', 'gaitpy', 'lumbar', 'waist', 'sensor', 'wearable', 'continuous wavelet', 'inverted pendulum', 'czech'],
                 classifiers=["Programming Language :: Python :: 3.6",
                              "License :: OSI Approved :: MIT License"],
                 license='MIT',
                 zip_safe=False,
                 install_requires=requirements(),
                 include_package_data=True)
