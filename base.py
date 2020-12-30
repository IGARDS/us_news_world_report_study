# Base Functionality for Reading and Processing
import os
from io import StringIO

import pandas as pd

data_dir = os.path.dirname(os.path.abspath(__file__))

def read_instance(name):
    df = pd.read_csv(f"{data_dir}/data/{name}.txt",header=None)
    df.columns = ['College','Overall Rank','Overall Score','Academic Reputation','Selectivity Rank','SAT (VM) 25-75th percentile','Percent freshmen top 10% HS Class','Acceptance','Faculty resource','Percent classes < 20 students','Percent classes >= 50 students','Student/Faculty ratio','Percent faculty full time','Graduation retention rank','Freshman retention','Financial resource rank','Alumni giving rank','Average Alumni giving rate']
    return df
