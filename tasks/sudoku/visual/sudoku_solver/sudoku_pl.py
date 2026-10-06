import os
from pathlib import Path

import paths

_SWIPL_ENV = str(paths.SWIPL_ENV)
_SWIPL_LIB = f"{_SWIPL_ENV}/lib/swipl/lib/x86_64-linux/libswipl.so"
_SWIPL_HOME = f"{_SWIPL_ENV}/lib/swipl"
if Path(_SWIPL_LIB).exists():
    os.environ.setdefault("LIBSWIPL_PATH", _SWIPL_LIB)
    os.environ.setdefault("SWI_HOME_DIR", _SWIPL_HOME)

from pyswip import Prolog
import numpy as np


PROLOG_RULES_PATH = Path(__file__).resolve().parent / "sudoku_prolog.pl"


def solve_sudoku(input_problem, prolog_instance=None):
    try:
        if not prolog_instance:
            prolog_instance = Prolog()
            prolog_instance.consult(str(PROLOG_RULES_PATH))
        input_type = type(input_problem)
        if input_type == list:
            input_problem = str(input_problem).replace('0','_')
        elif input_type == np.ndarray:
            input_problem = str(input_problem).replace('0','_').replace('\n','').replace(' ',',')
        
        solution_list =  list(prolog_instance.query("Rows=%s,sudoku(Rows)" % input_problem, maxresult=1))
        solution = []
        if len(solution_list)>0:
            solution = solution_list[0]["Rows"]
            if input_type == np.ndarray:
                solution = np.array(solution).astype(int)
        return solution
    except Exception as e:
        print('------------ prolog crashed')
        return []


def test1():
    prolog = Prolog()
    prolog.assertz("father(michael,john)")
    prolog.assertz("father(michael,gina)")
    list(prolog.query("father(michael,X)")) == [{'X': 'john'}, {'X': 'gina'}]
    for soln in prolog.query("father(X,Y)"):
        print(soln["X"], "is the father of", soln["Y"])


if __name__=='__main__':
    test1()

    
    

    