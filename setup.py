from setuptools import setup, find_packages

setup(
    name="deepmodels",
    version="0.1.0",  # Atualize a versão aqui
    author="GabrielFonsecaNunes",
    description="""DeepModels: Easy-to-Use Deep Learning Models for Time Series Analysis
DeepModels is a Python library that provides simple and intuitive access to advanced deep learning architectures such as\n 
LSTM, GRU, and Transformer, designed for time series forecasting and modeling. Inspired by the usability of statsmodels,\n
it offers a familiar interface with methods for fitting, predicting, and evaluating models. This library is ideal for users\n
who need powerful deep learning techniques without the complexity of low-level implementations.""",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GabrielFonsecaNunes/deep_learning_ts",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)