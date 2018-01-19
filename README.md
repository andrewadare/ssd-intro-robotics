This repo contains examples and demos for the Intro to Robotics class at the Boulder Hackerspace. More material will be added here as we continue through the class, so check back regularly.


Certain classes and functions are provided by a single importable module called `ssd_robotics`. To make this name visible to your python interpreter, run `pip3 install -e .` This will also ensure you have our code dependencies.

To test the installation, try `import ssd_robotics` in a python REPL session or notebook.


setup.py will then use setuptools module to retrieve and build the package as well as all dependent modules. Naturally, you must make sure that setuptools is available on your system. Without setuptools, you will encounter the error: "ImportError: No module named 'setuptools'".

PLEASE read below if you are using Linux!

Linux install instructions for apt-get distros:
--------------------

The most problems new Linux users will see are the following: dependency problems and installing the python GR package. 

GR is a universal framework for cross-platform visualization applications. It offers developers a compact, portable and consistent graphics library for their programs. Applications range from publication quality 2D graphs to the representation of complex 3D scenes.

Here is a short version of the install instructions for Linux.

1. Install Python3 with `apt-get install python3`
2. Install Python setuptools:

```
To install setuptools on Debian, Ubuntu or Mint:
$ sudo apt-get install python-setuptools

For Python 3.X applications, install python3-setuptools instead.
$ sudo apt-get install python3-setuptools 
```


3. Install OpenCV2 [here](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)


4. Follow the installation instructions for installing python-gr described [here](https://software.opensuse.org/download.html?project=science:gr-framework&package=python-gr).

5. Run these commands in a folder where you want to create a new Python virtual environment *IF* you didn't follow the opencv2 instructions.

`python3 -m venv some_name
source some_name/bin/activate`

Note: You will have to use the source command everytime you open a terminal to run our code.  This is because your system-level Python site-packages folder will not have our dependencies. Unless you add this to your bash rc file.

6. run `pip3 install -e .` in our ssd-intro-robotics folder to install our code dependencies.

7. Have Fun!

## Creating Virtual Environments

Section taken from https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments

Python “Virtual Environments” allow Python packages to be installed in an isolated location for a particular application, rather than being installed globally.

Imagine you have an application that needs version 1 of LibFoo, but another application requires version 2. How can you use both these applications? If you install everything into /usr/lib/python2.7/site-packages (or whatever your platform’s standard location is), it’s easy to end up in a situation where you unintentionally upgrade an application that shouldn’t be upgraded.

Or more generally, what if you want to install an application and leave it be? If an application works, any change in its libraries or the versions of those libraries can break the application.

Also, what if you can’t install packages into the global site-packages directory? For instance, on a shared host.

In all these cases, virtual environments can help you. They have their own installation directories and they don’t share libraries with other virtual environments.

Currently, there are two common tools for creating Python virtual environments:

- venv is available by default in Python 3.3 and later, and installs pip and setuptools into created virtual environments in Python 3.4 and later.
- virtualenv needs to be installed separately, but supports Python 2.6+ and Python 3.3+, and pip, setuptools and wheel are always installed into created virtual environments by default (regardless of Python version).

The basic usage is like so:

Using virtualenv:

```
virtualenv <DIR>
source <DIR>/bin/activate
```

Using venv:

```
python3 -m venv <DIR>
source <DIR>/bin/activate
```
For more information, see the virtualenv docs or the venv docs.

Managing multiple virtual environments directly can become tedious, so the dependency management tutorial introduces a higher level tool, Pipenv, that automatically manages a separate virtual environment for each project and application that you work on.
