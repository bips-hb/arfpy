import setuptools

setuptools.setup(
    name="arfpy",
    version = "0.1.1",
    author= "Kristin Blesch, Marvin N. Wright",
    author_email= "blesch@leibniz-bips.de",
    description="Adversarial random forests for density estimation and generative modeling",
    long_description="Adversarial random forests (ARFs) recursively partition data into fully factorized leaves, where features are jointly independent. The procedure is iterative, with alternating rounds of generation and discrimination. Data become increasingly realistic at each round, until original and synthetic samples can no longer be reliably distinguished. This is useful for several unsupervised learning tasks, such as density estimation and generative modeling. Methods for both are implemented in this package.",
        long_description_content_type="text/markdown",
    packages=['arfpy'],
    url='https://github.com/bips-hb/arfpy',
    license='MIT',
    install_requires= ['numpy>=1.20.3','pandas>=1.5','scikit-learn>=0.24','scipy>=1.4'],
                       
    classifiers=["Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Programming Language :: Python :: 3.10",
                 "Programming Language :: Python :: 3.11"]
)
