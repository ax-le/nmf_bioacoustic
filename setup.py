import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nmf_bioacoustic",
    version="0.1.1",
    author="Marmoret Axel",
    author_email="axel.marmoret@imt-atlantique.fr",
    description="Testing NMF for bioacoustic signals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.imt-atlantique.fr/a23marmo/nmf_bioacoustic",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3.8"
    ],
    license='BSD',
    install_requires=[
        'ipython',
        'librosa>=0.10.2.post1',
        'matplotlib',
        'mir_eval',
        'nn_fac>=0.3.1',
        'numpy >= 1.8.0',
        'pandas',
        'sacred',
        'scikit_learn',
        'scipy>=1.13.1',
        'setuptools',
        'tqdm',
    ]
)
