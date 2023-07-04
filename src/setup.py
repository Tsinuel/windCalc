from setuptools import setup, find_packages

setup(
    name='windCalc',
    version='1.0',
    packages=find_packages(),
    py_modules=['wind', 'foam', 'windCAD', 'windBLWT', 'windIO', 'windPlotting'],
    # package_dir={'wind': 'src/wind', 'foam': 'src/foam', 'windCAD': 'src/windCAD', 'windBLWT': 'src/windBLWT', 'windIO': 'src/windIO', 'windPlotting': 'src/windPlotting'},
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'shapely',
    ],
    # entry_points={
    #     'console_scripts': [
    #         'windCalc=windCalc.__main__:main',
    #     ],
    # },
    # package_data={
    #     'windCalc': ['data/*'],
    # },
    include_package_data=True,
    author='Tsinuel Geleta',
)