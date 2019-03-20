'''
Imports all the modules from subfolders
'''

import sys,os
import socket, re

# Input current folder's path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Input folder paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/" + "pyJets")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/" + "pyLandau")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/" + "pySlams")

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
