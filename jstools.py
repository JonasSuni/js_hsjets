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
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pySlams")

#Import modules

try:
    import jets
except ImportError as e:
    print("Note: Did not import jets module: ",e)

try:
    import landau
except ImportError as e:
    print("Note: Did not import landau module: ",e)

try:
    import slams
except ImportError as e:
    print("Note: Did not import slams module: ",e)    