from setuptools import setup
import tkinter
import sys
import os

# -- Replace “3.13” with your actual minor version if different.
PY_VER = "3.13"
TK_FRAMEWORK = f"/Library/Frameworks/Python.framework/Versions/3.13/Frameworks/Tk.framework"
TCL_FRAMEWORK = f"/Library/Frameworks/Python.framework/Versions/3.13/Frameworks/Tcl.framework"

APP = ['Stats_Calculator.py']
DATA_FILES = ['requirements.txt']  # any extra non-Python files you want bundled

OPTIONS = {
    # Include packages that your script uses
    'packages': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy'],
    # Force inclusion of the Tkinter backends
    'includes': [
        'tkinter',
        'tkinter.filedialog',
        'matplotlib.backends.backend_tkagg'
    ],
    # Tell py2app to bundle the Tcl + Tk frameworks from your Python-built location:
    'frameworks': [
        TCL_FRAMEWORK,
        TK_FRAMEWORK
    ],
    # For a Tkinter GUI, it’s usually best to disable argv_emulation:
    'argv_emulation': False,
    # You can set other plist keys here if you need them. Often not required for simple scripts.
    'plist': {
        # (If you ever need to override any Info.plist entries, you could do it here.)
    },
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

