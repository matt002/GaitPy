import setuptools

def requirements():
    with open('requirements.txt', "r") as fh:
        return [x for x in fh.read().split('\n') if x]

setuptools.setup(name='gaitpy',
                 version='1.0',
                 description='Read and process raw vertical accelerometry data from a lumbar sensor during gait; calculate clinical gait characteristics.',
                 url='http://github.com/PfizerRD/gaitpy',
                 packages=setuptools.find_packages(),
                 author='Matthew Czech',
                 author_email='czech1117@gmail.com',
                 keywords=['gait', 'lumbar', 'sensor', 'digital', 'wearable', 'python', 'inverted pendulum', 'czech'],
                 classifiers=["Programming Language :: Python :: 2.7",
                              "License :: OSI Approved :: MIT License"],
                 license='MIT',
                 zip_safe=False,
                 install_requires=requirements(),
                 include_package_data=True)