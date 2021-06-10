<<<<<<< HEAD
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # US News World Report Analysis

# %% slideshow={"slide_type": "skip"}
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %% slideshow={"slide_type": "skip"}
import pandas as pd
import numpy as np

# %% slideshow={"slide_type": "skip"}
=======
#!/usr/bin/env python
# coding: utf-8
# %%

# # US News World Report Analysis

# %%


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%


import pandas as pd
import numpy as np


# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv

from IPython.display import Image
def draw(A):
    return Image(A.draw(format='png', prog='dot'))


<<<<<<< HEAD
# %% slideshow={"slide_type": "skip"}
=======
# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
import sys
from pathlib import Path
home = str(Path.home())

<<<<<<< HEAD
sys.path.insert(0,"%s/ranking_toolbox"%home)
sys.path.insert(0,"%s/RPLib"%home)

# %% slideshow={"slide_type": "skip"}
import pyrankability

# %% slideshow={"slide_type": "skip"}
import pyrplib

# %% slideshow={"slide_type": "skip"}
years = [2002,2003,2004]

# %% slideshow={"slide_type": "skip"}
=======
sys.path.insert(0,"%s/rankability_toolbox_dev"%home)
sys.path.insert(0,"%s/RPLib"%home)


# %%


import pyrankability


# %%


import pyrplib


# %%


years = [2002,2003,2004]


# ## TODO
# Goal is to make this more general.
# 
# 1. Download the raw data (spreadsheets) from us news and world report. Hoping that has categories defined easily. Liberal arts colleges, etc. If not, we'll have do some of this ourselves or ...
#     i. Option 1 is to define the colleges
# 
# 2. Query is the set of important features in the dataset

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
colleges = """
Amherst College
Bowdoin College
Carleton College
Claremont McKenna College
Davidson College
Haverford College
Middlebury College
Pomona College
Swarthmore College
Wellesley College
Williams College
""".strip().split('\n')
colleges

<<<<<<< HEAD
# %% slideshow={"slide_type": "subslide"}
pd.Series(colleges)

# %% slideshow={"slide_type": "skip"}
=======

# %%


pd.Series(colleges)


# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
data = {}
for year in years:
    data[year] = pd.read_excel('data/USNews liberal arts 2002-2016 (1).xls',sheet_name=str(year))
    data[year]['School Name'] = data[year]['School Name'].str.replace('!','')
    if 'State' in data[year].columns:
        data[year]['State'] = data[year]['State'].str.replace('\(','').str.replace('\)','')
    df = pd.DataFrame(list(data[year]['SAT/ACT 25th-75th Percentile'].str.split('-')),columns=['SAT/ACT 25th Percentile','SAT/ACT 75th Percentile'])
    data[year] = pd.concat([data[year],df],axis=1)
    data[year] = data[year].infer_objects()
    data[year]['SAT/ACT 25th-75th Percentile Mean'] = (data[year]['SAT/ACT 25th Percentile'].astype(int)+data[year]['SAT/ACT 75th Percentile'].astype(int))/2


# %%
<<<<<<< HEAD
=======


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
data2 = {}
for year in years:
    data2[year] = data[year].set_index('School Name')
    data2[year] = data2[year].loc[colleges]

<<<<<<< HEAD
# %% [markdown]
# ### Now select the year you want to process

# %%
year = 2002

# %% slideshow={"slide_type": "skip"}
student_columns = {}
parent_columns = {}

# %% slideshow={"slide_type": "subslide"}
=======

# ### Now select the year you want to process

# %%


year = 2002


# %%


student_columns = {}
parent_columns = {}


# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
columns = list(data[year].columns)
student_columns[year] = [columns[13],columns[16],columns[22],columns[9],columns[11]]
student_columns[year]

<<<<<<< HEAD
# %% slideshow={"slide_type": "subslide"}
parent_columns[year] = [columns[6],columns[9],columns[11],columns[13],columns[16]]
parent_columns[year]

# %%
set(parent_columns[year]).union(student_columns[year])

# %% [markdown] slideshow={"slide_type": "skip"}
# Do we have all the colleges? 

# %% slideshow={"slide_type": "skip"}
assert sum(data[year]['School Name'].isin(colleges)) == len(colleges)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Transform all the columns so they are the same direction as Final Rank column

# %% slideshow={"slide_type": "skip"}
=======

# %%


parent_columns[year] = [columns[6],columns[9],columns[11],columns[13],columns[16]]
parent_columns[year]


# %%


set(parent_columns[year]).union(student_columns[year])


# Do we have all the colleges? 

# %%


assert sum(data[year]['School Name'].isin(colleges)) == len(colleges)


# ## Transform all the columns so they are the same direction as Final Rank column

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
from sklearn.pipeline import Pipeline
Ds = pd.DataFrame(columns=['Year','Group','D']).set_index(['Year','Group'])
parent_pipe = Pipeline([('fix_sign', pyrplib.transformers.ColumnDirectionTransformer('Final Rank')),
                        ('count_transformer', pyrplib.transformers.ColumnCountTransformer(parent_columns[year]))
                ])

student_pipe = Pipeline([('fix_sign', pyrplib.transformers.ColumnDirectionTransformer('Final Rank')),
                        ('count_transformer', pyrplib.transformers.ColumnCountTransformer(student_columns[year]))
                ])

both_columns = list(set(parent_columns[year]+student_columns[year]))

both_pipe = Pipeline([('fix_sign', pyrplib.transformers.ColumnDirectionTransformer('Final Rank')),
                        ('count_transformer', pyrplib.transformers.ColumnCountTransformer(both_columns))
                ])

parent_pipe.fit(data2[year])
D = parent_pipe.transform(data2[year])
Ds = Ds.append(pd.Series([D],index=['D'],name=(year,'Parent')))

student_pipe.fit(data2[year])
D = student_pipe.transform(data2[year])
Ds = Ds.append(pd.Series([D],index=['D'],name=(year,'Student')))

both_pipe.fit(data2[year])
D = both_pipe.transform(data2[year])
Ds = Ds.append(pd.Series([D],index=['D'],name=(year,'Both')))

<<<<<<< HEAD
# %% slideshow={"slide_type": "subslide"}
Ds.loc[(year,'Both'),'D']

# %% slideshow={"slide_type": "subslide"}
Ds.loc[(year,'Student'),'D']

# %% slideshow={"slide_type": "subslide"}
Ds.loc[(year,'Parent'),'D']


# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Nearest and farthest from Centroid for combined columns, parent, and student

# %% slideshow={"slide_type": "skip"}
=======

# %%


Ds.loc[(year,'Both'),'D']


# %%


Ds.loc[(year,'Student'),'D']


# %%


Ds.loc[(year,'Parent'),'D']


# ## Nearest and farthest from Centroid for combined columns, parent, and student

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
class Details:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
for index in Ds.index:
    D = Ds.loc[index,'D']
    delta_cont, details_cont = pyrankability.rank.solve(D,method='lop',cont=True)
    delta_bin, details_bin = pyrankability.rank.solve(D,method='lop',cont=False)
    Ds.loc[index,'delta_cont'] = delta_cont
    Ds.loc[index,'delta_bin'] = delta_bin
    Ds.loc[index,'details_cont'] = Details(**details_cont)
    Ds.loc[index,'details_bin'] = Details(**details_bin)
    _, details_fixed_cont_x_maximize = pyrankability.search.solve_fixed_cont_x(D,details_bin['obj'],details_cont['x'],method='lop',minimize=False)
    _, details_fixed_cont_x_minimize = pyrankability.search.solve_fixed_cont_x(D,details_bin['obj'],details_cont['x'],method='lop',minimize=True)    
    Ds.loc[index,'details_fixed_cont_x_maximize'] = Details(**details_fixed_cont_x_maximize)
    Ds.loc[index,'details_fixed_cont_x_minimize'] = Details(**details_fixed_cont_x_minimize)
    _, details_farthest_pair_maximize = pyrankability.search.solve_pair(D,method='lop',minimize=False)#,min_ndis=None,max_ndis=None,tau_range=None,lazy=False,verbose=True)
    Ds.loc[index,'details_pair_maximize'] = Details(**details_farthest_pair_maximize)
    _, details_farthest_pair_minimize = pyrankability.search.solve_pair(D,method='lop',minimize=True, min_ndis=1)#,min_ndis=None,max_ndis=None,tau_range=None,lazy=False,verbose=True)
    Ds.loc[index,'details_pair_minimize'] = Details(**details_farthest_pair_minimize)


# %%
<<<<<<< HEAD
=======


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
closest_obj = Ds.loc[(year,'Both'),'details_fixed_cont_x_minimize'].obj
farthest_obj = Ds.loc[(year,'Both'),'details_fixed_cont_x_maximize'].obj
closest_obj,farthest_obj

<<<<<<< HEAD
# %%
pair_closest_obj = Ds.loc[(year,'Both'),'details_pair_maximize'].obj
pair_farthest_obj = Ds.loc[(year,'Both'),'details_pair_minimize'].obj
pair_closest_obj,pair_farthest_obj


# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Look at the continuous results

# %% slideshow={"slide_type": "skip"}
=======

# ## Look at the continuous results

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
def plot_xstars(Ds,show_score_xstar2_func=lambda xstars: pyrankability.plot.show_score_xstar2(xstars,
                                                                     group_label="Group",width=300,height=300,
                                                                     columns=1,resolve_scale=True)):
    label = "A"
    xstars = {}
    for index in Ds.index:
        D = Ds.loc[index,'D']
        xstar = pd.DataFrame(Ds.loc[index,'details_cont'].x,index=D.index,columns=D.columns)
        xstars["%s. %s"%(label,", ".join(str(v) for v in index[::-1]))] = xstar
        label = chr(ord(label)+1)
    g,score_df,ordered_xstars = show_score_xstar2_func(xstars)
    return g,score_df,ordered_xstars


<<<<<<< HEAD
# %% slideshow={"slide_type": "subslide"}
g,_,_ = plot_xstars(Ds)
g

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Compare centroids for max and min for each group

# %% slideshow={"slide_type": "fragment"}
pd.Series(Ds.loc[(year,'Both'),'D'].index)

# %% slideshow={"slide_type": "skip"}
colleges

# %% slideshow={"slide_type": "skip"}
=======
# %%


g,_,_ = plot_xstars(Ds)
g


# ## Compare centroids for max and min for each group

# %%


pd.Series(Ds.loc[(year,'Both'),'D'].index)


# %%


colleges


# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
Ds.loc[(year,'Parent'),'details_fixed_cont_x_minimize'].perm


# %%
<<<<<<< HEAD
=======


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
def perm_to_series(D,perm,name):
    return pd.Series(list(D.index[list(perm)]),name=name)


<<<<<<< HEAD
# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Parent

# %% slideshow={"slide_type": "fragment"}
=======
# ### Parent

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
A = perm_to_series(Ds.loc[(year,'Parent'),'D'],Ds.loc[(year,'Parent'),'details_fixed_cont_x_minimize'].perm,'Closest')
B = perm_to_series(Ds.loc[(year,'Parent'),'D'],Ds.loc[(year,'Parent'),'details_fixed_cont_x_maximize'].perm,'Farthest')
pyrankability.plot.spider2(A,B,file="results/parent_fixed_cont_x_minimize_maximize.png",xmult = 1.7,ymult=1.1)

<<<<<<< HEAD
# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Student

# %% slideshow={"slide_type": "fragment"}
=======

# ### Student

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
A = perm_to_series(Ds.loc[(year,'Student'),'D'],Ds.loc[(year,'Student'),'details_fixed_cont_x_minimize'].perm,'Closest')
B = perm_to_series(Ds.loc[(year,'Student'),'D'],Ds.loc[(year,'Student'),'details_fixed_cont_x_maximize'].perm,'Farthest')
pyrankability.plot.spider2(A,B,file="results/student_fixed_cont_x_minimize_maximize.png")

<<<<<<< HEAD
# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Both

# %% slideshow={"slide_type": "fragment"}
=======

# ### Both

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
A = perm_to_series(Ds.loc[(year,'Both'),'D'],Ds.loc[(year,'Both'),'details_fixed_cont_x_minimize'].perm,'Closest')
B = perm_to_series(Ds.loc[(year,'Both'),'D'],Ds.loc[(year,'Both'),'details_fixed_cont_x_maximize'].perm,'Farthest')
pyrankability.plot.spider2(A,B,file="results/both_fixed_cont_x_minimize_maximize.png")

<<<<<<< HEAD
# %%
A

# %%
list(A)

# %%
B

# %%
list(B)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Now let's consider the centroid to a ranking of interest (Final rank)

# %% slideshow={"slide_type": "skip"}
data2[year]['Final Rank'].argsort()

# %% slideshow={"slide_type": "skip"}
data2[year]['Final Rank']

# %% slideshow={"slide_type": "skip"}
data2[year]['Final Rank'].iloc[data2[year]['Final Rank'].argsort()]

# %% slideshow={"slide_type": "subslide"}
perm_final_rank = data2[year]['Final Rank'].argsort()
x_final_rank = pyrankability.common.perm_to_x(perm_final_rank)

# %% slideshow={"slide_type": "subslide"}
=======

# %%


A


# %%


list(A)


# %%


B


# %%


list(B)


# ### Now let's consider the centroid to a ranking of interest (Final rank)

# %%


data2[year]['Final Rank'].argsort()


# %%


data2[year]['Final Rank']


# %%


data2[year]['Final Rank'].iloc[data2[year]['Final Rank'].argsort()]


# %%


perm_final_rank = data2[year]['Final Rank'].argsort()
x_final_rank = pyrankability.common.perm_to_x(perm_final_rank)


# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
index = (year,'Student')
D = Ds.loc[index,'D']
delta_bin, details_bin = pyrankability.rank.solve(D,method='lop',cont=False)
_, details_fixed_binary_x_maximize = pyrankability.search.solve_fixed_binary_x(D,details_bin['obj'],x_final_rank,method='lop',minimize=False)
_, details_fixed_binary_x_minimize = pyrankability.search.solve_fixed_binary_x(D,details_bin['obj'],x_final_rank,method='lop',minimize=True)    

<<<<<<< HEAD
# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Maximize (farthest)

# %% slideshow={"slide_type": "fragment"}
=======

# ### Maximize (farthest)

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
A = perm_to_series(D,perm_final_rank,'Final Rank')
B = perm_to_series(D,details_fixed_binary_x_maximize['perm'],'Farthest')
pyrankability.plot.spider2(A,B,file="results/student_details_fixed_binary_x_maximize.png")

<<<<<<< HEAD
# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Minimize (closest)

# %% slideshow={"slide_type": "fragment"}
=======

# ### Minimize (closest)

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
A = perm_to_series(D,perm_final_rank,'Final Rank')
B = perm_to_series(D,details_fixed_binary_x_minimize['perm'],'Closest')
pyrankability.plot.spider2(A,B,file="results/student_details_fixed_binary_x_minimize.png")

<<<<<<< HEAD
# %% [markdown]
# ### Farthest pair

# %%
=======

# ### Farthest pair

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
index = (year,'Parent')
D = Ds.loc[index,'D']
obj, farthest_pair_details = pyrankability.search.solve_pair(D,method='lop',minimize=False)#,min_ndis=None,max_ndis=None,tau_range=None,lazy=False,verbose=True)

A = perm_to_series(D,farthest_pair_details['perm_x'],'perm_x')
B = perm_to_series(D,farthest_pair_details['perm_y'],'perm_y')
pyrankability.plot.spider2(A,B,file=f'results/farthest_pair_{index}.png')

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
index = (year,'Student')
D = Ds.loc[index,'D']
obj, farthest_pair_details = pyrankability.search.solve_pair(D,method='lop',minimize=False)#,min_ndis=None,max_ndis=None,tau_range=None,lazy=False,verbose=True)

A = perm_to_series(D,farthest_pair_details['perm_x'],'perm_x')
B = perm_to_series(D,farthest_pair_details['perm_y'],'perm_y')
pyrankability.plot.spider2(A,B,file=f'results/farthest_pair_{index}.png')

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
index = (year,'Both')
D = Ds.loc[index,'D']
obj, farthest_pair_details = pyrankability.search.solve_pair(D,method='lop',minimize=False)#,min_ndis=None,max_ndis=None,tau_range=None,lazy=False,verbose=True)

A = perm_to_series(D,farthest_pair_details['perm_x'],'perm_x')
B = perm_to_series(D,farthest_pair_details['perm_y'],'perm_y')
pyrankability.plot.spider2(A,B,file=f'results/farthest_pair_{index}.png')

<<<<<<< HEAD
# %% [markdown]
# ### Nearest pair

# %%
=======

# ### Nearest pair

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
index = (year,'Both')
D = Ds.loc[index,'D']
obj, nearest_pair_details = pyrankability.search.solve_pair(D,method='lop',minimize=True,min_ndis=1)

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
A = perm_to_series(D,nearest_pair_details['perm_x'],'perm_x')
B = perm_to_series(D,nearest_pair_details['perm_y'],'perm_y')
pyrankability.plot.spider2(A,B,file=f'results/nearest_pair_{index}.png')

<<<<<<< HEAD
# %% [markdown]
# ## Two datasets farthest

# %%
=======

# ## Two datasets farthest

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
index = (year,'Student')
D = Ds.loc[index,'D']
index2 = (year,'Parent')
D2 = Ds.loc[index2,'D']
obj, farthest_pair_details_D_D2 = pyrankability.search.solve_pair(D,D2=D2,method='lop',minimize=False)
obj

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
A = perm_to_series(D,farthest_pair_details_D_D2['perm_x'],'perm_x')
B = perm_to_series(D,farthest_pair_details_D_D2['perm_y'],'perm_y')
pyrankability.plot.spider2(A,B,file=f'results/farthest_pair_{index}_{index2}.png')

<<<<<<< HEAD
# %% [markdown]
# ### Student and parent nearest

# %%
=======

# ### Student and parent nearest

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
index = (year,'Student')
D = Ds.loc[index,'D']
index2 = (year,'Parent')
D2 = Ds.loc[index2,'D']
obj, nearest_pair_details_D_D2 = pyrankability.search.solve_pair(D,D2=D2,method='lop',minimize=True)

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
A = perm_to_series(D,nearest_pair_details_D_D2['perm_x'],'perm_x')
B = perm_to_series(D,nearest_pair_details_D_D2['perm_y'],'perm_y')
pyrankability.plot.spider2(A,B,file=f'results/nearest_pair_{index}_{index2}.png')

<<<<<<< HEAD
# %%
import joblib
joblib.dump(Ds,"results/Ds.joblib.z");

# %%
D = Ds.loc[(2002,'Both'),'D']
D

# %%
=======

# %%


import joblib
joblib.dump(Ds,"results/Ds.joblib.z");


# %%


D = Ds.loc[(2002,'Both'),'D']
D


# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
_,details_lop_with_models = pyrankability.rank.solve(D,method='lop',include_model=True)
model = details_lop_with_models['model']
model_file = pyrankability.common.write_model(model)
solution_file = model_file + ".solutions"
<<<<<<< HEAD
# !sed -i '/^OBJSENS/d' $model_file
# !$home/ranking_toolbox/collect.sh "$model_file" "$solution_file"
=======
get_ipython().system("sed -i '/^OBJSENS/d' $model_file")
get_ipython().system('$home/rankability_toolbox_dev/collect.sh "$model_file" "$solution_file"')
>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7

opt_k = details_lop_with_models['obj']

solutions = pd.read_csv(solution_file,sep=', ')
x_columns = solutions.columns[1:-1]
xs = []
a,b,c = 1,1,-2*len(x_columns)
n = int((-b + np.sqrt(b**2 - 4*a*c))/(2*a) + 1)
xstar = np.zeros((n,n))
objs = []
s = 0
for k in range(solutions.shape[0]):
    x = np.zeros((n,n))
    for c in x_columns:
        ij_str = c.replace("x(","").replace(")","")
        i,j = ij_str.split(",")
        i,j = int(i),int(j)
        x[i,j] = solutions.loc[k,c]
        x[j,i] = 1 - x[i,j]
    obj = np.sum(np.sum(D*x))
    xs.append(x)
    objs.append(obj)
    error = obj - opt_k
    xstar += x
xstar = xstar/solutions.shape[0]

perms = []
for x in xs:
    r = np.sum(x,axis=0)
    perm = np.argsort(r)
    perms.append(perm)

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
from scipy import stats
taus = {} #np.zeros((len(perms),len(perms)))
for i in range(len(perms)):
    key = tuple(perms[i])
    taus[key] = []
    for j in range(len(perms)):
        tau,pvalue = stats.kendalltau(np.argsort(perms[i]), np.argsort(perms[j]))
        taus[key].append(tau)

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
taus_df = pd.DataFrame(columns=range(len(perms)))
for i in range(len(perms)):
    key = tuple(perms[i])
    taus_df = taus_df.append(pd.Series(taus[key],name=key))

<<<<<<< HEAD
# %%
taus_df

# %%
taus_df.values[np.where(taus_df == 1)] = np.NaN

# %%
closest = Ds.loc[(year,'Both'),'details_fixed_cont_x_minimize'].perm
farthest = Ds.loc[(year,'Both'),'details_fixed_cont_x_maximize'].perm

# %%
=======

# %%


taus_df


# %%


taus_df.values[np.where(taus_df == 1)] = np.NaN


# %%


closest = Ds.loc[(year,'Both'),'details_fixed_cont_x_minimize'].perm
farthest = Ds.loc[(year,'Both'),'details_fixed_cont_x_maximize'].perm


# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
source = taus_df.loc[[closest,farthest]]
source.index=['Closest','Farthest']
source = source.reset_index().melt(id_vars=['index'])
source.columns = ["Solution","To","Tau"]
source['pct'] = 1/len(source)
source

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
import altair as alt 
source2 = taus_df.loc[[closest,farthest]]
source2.index=['Closest','Farthest']
source2.T

g = alt.Chart(source2.T).mark_point().encode(
        x=alt.X('Closest:Q'),
        y=alt.Y('Farthest:Q')
    )
g

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
import altair as alt
g = alt.Chart(source).mark_area(
        opacity=0.6,
        interpolate='step'
    ).encode(
        x=alt.X('Tau:Q', bin=alt.Bin(maxbins=100)),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'),stack=None,title='Percentage'),
        color=alt.Color('Solution:N')
    )
g

<<<<<<< HEAD
# %%
=======

# %%


>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
import altair as alt
g = alt.Chart(source).mark_area(
        opacity=0.6,
        interpolate='step'
    ).encode(
        x=alt.X('Tau:Q', bin=alt.Bin(maxbins=100)),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'),stack=None,title='Percentage'),
        color=alt.Color('Solution:N')
    )
g

<<<<<<< HEAD
# %%
len(perms)

# %%
=======

# %%


len(perms)


# %%




>>>>>>> af67ecf1028d33489df168b24b9c2a4c0fe76dd7
