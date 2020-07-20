import setuptools

setuptools.setup(
        name='trainer',
        version='0.1',
        packages=setuptools.find_packages(),
        install_requires=['scipy>1.4'],
        include_package_data=True,
        description='Training application'
)
