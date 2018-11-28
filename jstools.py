'''
Imports all the modules from subfolders
'''

import filemanagement
import socket, re

# Input current folder's path
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)))
# Input folder paths
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyJets")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyLandau")

#Import modules

try:
    import jets
except ImportError as e:
    print("Note: Did not import jets module: ",e)

try:
    import landau
except ImportError as e:
    print("Note: Did not import jets module: ",e)