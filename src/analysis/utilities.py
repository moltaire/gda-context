import os


def makeDirIfNeeded(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
