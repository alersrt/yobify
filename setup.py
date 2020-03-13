from setuptools import setup, find_packages

setup(
    name="YOUBIFY",
    version="0.1",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
      'flask',
      'gunicorn',
      'docutils',
      'jupyter',
      'matplotlib',
      'pandas',
      'numpy',
      'sklearn',
      'keras',
      'tensorflow-cpu',
      'opencv-python',
      'mtcnn',
      'moviepy'
    ],
)
