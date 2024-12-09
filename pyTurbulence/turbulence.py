"""Imports all the modules in the pyTurbulence folder"""

try:
    import plot_1d
except ImportError as e:
    print("Note: Did not import plot_1d module: ", e)