# Import C bindings
from _microml import *

# Submodules
from . import utils
from .preprocessing import StandardScaler, MinMaxScaler
from .metrics import accuracy_score, confusion_matrix # Import all C module symbols into top-level microml namespace



