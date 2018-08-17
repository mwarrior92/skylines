from helpers import format_dirpath, mydir
import matplotlib.pyplot as plt
import json

################### SET UP FILE I/O ######################
topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardataf = format_dirpath(topdir+'data/parse_hars/')+'getlists.json'
label = 'query_experiment'
datadir = format_dirpath(topdir+"data/"+label)
platform = "ripe_atlas"

