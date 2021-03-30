import setuptools

setuptools.setup(
    name="DeezyMatch",
    version="1.2.3",
    description="A Flexible Deep Learning Approach to Fuzzy String Matching",
    author=u"The LwM Development Team",
    #author_email="",
    license="MIT License",
    keywords=["Fuzzy String Matching", "Deep Learning", "NLP", "Natural Language Processing", "living with machines"],
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    url="https://github.com/Living-with-machines/DeezyMatch",
    download_url="https://github.com/Living-with-machines/DeezyMatch/archive/master.zip",
    packages = setuptools.find_packages(),
    include_package_data = True,
    platforms="OS Independent",
    python_requires='>=3.7',
    install_requires=[
        "torch==1.5.0",
        "torchvision==0.6.0",
        "ipywidgets==7.5.1",
        "PyYAML==5.4",
        "scikit-learn==0.23.1",
        "pandas==1.0.3",
        "faiss-cpu==1.6.3",
        "tqdm==4.46.0",
        "tensorboard==2.2.2",
        "matplotlib==3.2.1",
        "jupyter-client==6.1.5",
        "jupyter-core==4.6.3"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],

    entry_points={
        'console_scripts': [
            'DeezyMatch = DeezyMatch.DeezyMatch:main',
        ],
    }
)
