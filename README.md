------------------------------------
Securing Data Analytics on SGX with Randomization
------------------------------------
Details can be found in here.

Project:
1. App/App.cpp contains the untrusted code.
2. Enclave/Analytics/ contain the data analytics used in various experiments in the paper.

Settings (in App/App.cpp):
1. basedir: Directory of input data files.
2. Experiments settings on chunk/minibatch size, proportion of dummy data instances, number of clusters in K-Means clustering.
3. Class label prediction analytics include Decision Tree, Naive Bayes and K-means. Each classifier can be run by choosing appropriate code, as given in App.cpp under ocall_manager function.


------------------------------------
How to Build/Execute the Code
------------------------------------
1. Install Intel(R) SGX SDK for Linux* OS
2. Specify data directory in basedir of App/App.cpp.
3. Also specify appropriate settings in App/App.cpp file.
2. Build the project with the prepared Makefile:
    a. Hardware Mode, Debug build:
        $ make SGX_MODE=HW SGX_DEBUG=1
    b. Hardware Mode, Pre-release build:
        $ make SGX_MODE=HW SGX_PRERELEASE=1
    c. Hardware Mode, Release build:
        $ make SGX_MODE=HW
    d. Simulation Mode, Debug build:
        $ make SGX_DEBUG=1
    e. Simulation Mode, Pre-release build:
        $ make SGX_PRERELEASE=1
    f. Simulation Mode, Release build:
        $ make
3. Execute the binary directly:
    $ ./app

