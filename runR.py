import subprocess

# Calls R script 
res = subprocess.call("Rscript script.R", shell=True)
res


from rpy2 import robjects

pi = robjects.r['pi']
pi

type(pi)

robjects.r('''
    add_nums <- function(x, y) {
        return(x + y)
    }
    
    print(add_nums(x = 5, y = 10))
    print(add_nums(x = 10, y = 20))
''')


import rpy2.robjects.packages as rpackages

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

utils.install_packages('dplyr')

from rpy2.robjects.packages import importr, data

datasets = importr('datasets')
mtcars = data(datasets).fetch('mtcars')['mtcars']
mtcars