#####
#take funpack chunk and further filter.  Then run some stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.stats import chi2_contingency
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from mlxtend.plotting import scatterplotmatrix
def density_scatter( x , y, s,j, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    ax.set_xlabel(s,fontsize=18)
    ax.set_ylabel(j,fontsize=18)

    sns.regplot(x, y, ci=95,scatter=False)


    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    x = sm.add_constant(x)
    model = sm.OLS(y, x)

    results = model.fit()
    print(results.params)

    return ax
def regress_component(vars,c,abphenonmf):
    x = abphenonmf[vars]
    y = abphenonmf[c]
    # with sklearn
    print("*************COMPONENT ",c," *****************")

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit(cov_type='HC3')
    pvals = model.pvalues

    #result = model_ols.fit(cov_type='HC3')
    predictions = model.predict(x)
    print_model = model.summary()
    print(print_model)
    print("*************END ", c, " *****************")
    return pvals

#import files
k40a=pd.read_csv("/Users/petralenzini/chpc3/headmotion/briefReport2023/HeadMotionVars.csv",low_memory=False)
k40b=pd.read_csv("/Users/petralenzini/chpc3/headmotion/briefReport2023/structuralheadmotion.csv",low_memory=False)
IDP_1=pd.read_csv("/Users/petralenzini/chpc3/headmotion/briefReport2023/structuralIDPs_ses-01_1000.csv")
#IDP_2=pd.read_csv("/Users/petralenzini/chpc3/headmotion/briefReport2023/structuralIDPs_ses-02.csv",low_memory=False)
#IDP=pd.merge(IDP_1,IDP_2,on='eid',how='outer')
IDP_header=pd.read_csv("/Users/petralenzini/chpc3/headmotion/briefReport2023/T1.DWI.SWI.IDP_annot.csv",low_memory=False)

#merge phenotypes and identify vars containing instance 3 stuff
k40=pd.merge(k40a,k40b,on='eid',how='left')
subsetcolumns=[x for x in list(k40.columns) if "-3." not in x]

#grab any of the ICD 10 code variables and create the ICD dataframe manipulation
subsetICD=[x for x in subsetcolumns if "41270-" in x]
ICD=k40[['eid']+subsetICD]
justphenos=[x for x in subsetcolumns if "41270-" not in x]

G37=pd.DataFrame()
for g in range(0,10):
    G=ICD[ICD.isin(['G37'+str(g)]).any(axis=1)].copy()
    G37=pd.concat([G37,G.eid])
G37=G37.drop_duplicates()

G36=pd.DataFrame()
for g6 in range(0,10):
    G6=ICD[ICD.isin(['G36'+str(g6)]).any(axis=1)].copy()
    G36=pd.concat([G36,G6.eid])
G36=G36.drop_duplicates()

G35=pd.DataFrame()
for g5 in range(0,10):
    G5=ICD[ICD.isin(['G35'+str(g5)]).any(axis=1)].copy()
    G35=pd.concat([G35,G5.eid])
G35=G35.drop_duplicates()

G35_37=pd.concat([G35,G36,G37]).drop_duplicates()
G35_37['G35_37']=1
G35_37.columns=['eid','G35_37']
G35_37['eid']=G35_37['eid'].astype(int)

k40slim=pd.merge(k40[justphenos],G35_37,on='eid',how='left')
k40slim.loc[~(k40slim.G35_37==1),'G35_37']=0

sex='31-0.0'
age='21003-2.0'
site='54-2.0'
ICV='26521-2.0'
strucmotion='24419-2.0'
restmotion='25741-2.0'
taskmotion='25742-2.0'
dropmotion=[restmotion,ICV]

#merge in IDPs
k40slim=pd.merge(k40slim,IDP_1.drop(columns=dropmotion),on='eid',how='inner')

# Calculate RDS
k40slim['RDS'] = k40slim['2050-2.0'] + k40slim['2060-2.0'] + k40slim['2070-2.0'] + k40slim['2080-2.0']
k40slim['RDS_BIN'] = ''
k40slim.loc[k40slim.RDS <= 5, 'RDS_BIN'] = 0
k40slim.loc[(k40slim.RDS > 5), 'RDS_BIN'] = 1
#if any of the vars is < 0
k40slim.loc[((k40slim['2050-2.0']<0) | (k40slim['2060-2.0']<0) | (k40slim['2070-2.0']<0) | (k40slim['2080-2.0']<0)),'RDS']=np.nan
k40slim.loc[((k40slim['2050-2.0']<0) | (k40slim['2060-2.0']<0) | (k40slim['2070-2.0']<0) | (k40slim['2080-2.0']<0)),'RDS_BIN']=np.nan


k=k40slim.loc[k40slim[strucmotion].isnull()==False]
#k=k.loc[k[restmotion].isnull()==False]
#k=k.loc[k[taskmotion].isnull()==False]
#k=k.loc[k.RDS.isnull()==False].copy()
k=k40slim.copy()

# Drop missings
for var in [site, ICV, sex, age, 'RDS',strucmotion,restmotion,taskmotion]:
    k = k.loc[~(k[var].isnull() == True)].copy()

k['site5']=0
k['site6']=0
k.loc[k[site]==11026,'site6']=1
k.loc[k[site]==11025,'site5']=1

# distribution of rest motion vs structural headmotion


###########################################
# fig violin plots of motion in time
# TO DO
###########################################

# now construct loop parameters
#CONSIDER AGE/SEX/SITE/ICV matched samples instead?


# Linear model
motionlist=[strucmotion,restmotion,taskmotion]
# strucmotion is f(affect)
affectlist=['RDS','G35_37']
covarsimple0=['site5','site6',ICV, sex, age]
covarsimple1=['site5','site6',ICV, sex, age,strucmotion]
covarsimple1b=['site5','site6',ICV, sex, age,restmotion]
covarsimple2=['site5','site6',ICV, sex, age,strucmotion,restmotion]
covarsimple3=['site5','site6',ICV, sex, age,strucmotion,restmotion,taskmotion]

IDParea=[str(x)+"-2.0" for x in list(IDP_header.loc[IDP_header.description.str.upper().str.contains('AREA')].variable)]
IDPvol=[str(x)+"-2.0" for x in list(IDP_header.loc[IDP_header.description.str.upper().str.contains('VOLUME')].variable)]
IDPthick=[str(x)+"-2.0" for x in list(IDP_header.loc[IDP_header.description.str.upper().str.contains('THICK')].variable)]

# restmotion is f(affect)
# IDP is f(strucmotion)
# IDP is f(restmotion)
# IDP is f(affect)

# IDP is f(affect + structural headmotion + ASS) [manhattan groups by IDP type]
# IDP is f(affect + structural headmotion + ASS + resting state headmotion + task headmotion)
# p-val of affect var as function of model complexity
# p-val of headmotion var


x=np.array(k[strucmotion])
y=np.array(k[restmotion])
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=.5)
plt.xlabel("Measure of head motion in T1 structural image")
plt.ylabel("resting state head motion, mm")
plt.show()
k[strucmotion].hist(bins=50)
k.shape
plt.xlabel("Measure of head motion in T1 structural image")
plt.show()
k[restmotion].hist(bins=50)
k.shape
plt.xlabel("resting state head motion, mm")
plt.show()

#univariate
n=k.copy()
#print(n.shape)
#for i in ['RDS',strucmotion, restmotion]:
#    n[i] = stats.zscore(n[i])


Pvals=pd.DataFrame()
Manhattan=pd.DataFrame()
for covars in [[],covarsimple0,covarsimple1,covarsimple1b,covarsimple2]:
    for xx in ['RDS','G35_37']:
        for yy in IDParea:
            n=n.loc[n[yy].isnull()==False]
            pvals = pd.DataFrame(regress_component([xx]+covars,[yy], n)).transpose()
            pvals['x']=xx
            pvals['covars']=str(covars)
            pvals['y']=yy
            pvals=pvals.rename(columns={xx:'P-value for x'})
            Pvals=pd.concat([Pvals,pvals],axis=0)
Pvals['IDP type']='Area'
Manhattan=pd.concat([Manhattan,Pvals],axis=0)

Pvals=pd.DataFrame()
for covars in [[],covarsimple0,covarsimple1,covarsimple1b,covarsimple2]:
    for xx in ['RDS','G35_37']:
        for yy in IDPvol:
            n=n.loc[n[yy].isnull()==False]
            pvals = pd.DataFrame(regress_component([xx]+covars,[yy], n)).transpose()
            pvals['x']=xx
            pvals['covars']=str(covars)
            pvals['y']=yy
            pvals=pvals.rename(columns={xx:'P-value for x'})
            Pvals=pd.concat([Pvals,pvals],axis=0)
Pvals['IDP type']='Volume'
Pvals.to_csv("VolumePvalues26Jul2023.csv",index=False)
Manhattan=pd.concat([Manhattan,Pvals],axis=0)
Manhattan.to_csv("Manhattan.csv",index=False)

Pvals=pd.DataFrame()
for covars in [[],covarsimple0,covarsimple1,covarsimple1b,covarsimple2]:
    for xx in ['RDS','G35_37']:
        for yy in IDPthick:
            n=n.loc[n[yy].isnull()==False]
            pvals = pd.DataFrame(regress_component([xx]+covars,[yy], n)).transpose()
            pvals['x']=xx
            pvals['covars']=str(covars)
            pvals['y']=yy
            pvals=pvals.rename(columns={xx:'P-value for x'})
            Pvals=pd.concat([Pvals,pvals],axis=0)
Pvals['IDP type']='Thickness'
Pvals.to_csv("ThicknessPvalues26Jul2023.csv",index=False)
Manhattan=pd.concat([Manhattan,Pvals],axis=0)

Pvals=pd.read_csv("ThicknessPvalues26Jul2023.csv")
#Make Manhattan
for affect in affectlist:
    for group in ['Volume']:#['Thickness','Volume',]:
        try:
            subset=Pvals.loc[(Pvals['IDP type']==group) & (Pvals.x==affect)]
            subset=subset.reset_index().drop(columns=['index'])
            fig, ax = plt.subplots()
            a = np.array(subset['P-value for x'])
            nlogp = -1 * np.log10(a)
            x = np.array(subset.reset_index()['index'])
            colors = {"[]":'red', "['site5', 'site6', '26521-2.0', '31-0.0', '21003-2.0']":'orange', "['site5', 'site6', '26521-2.0', '31-0.0', '21003-2.0', '25741-2.0']":'yellow', "['site5', 'site6', '26521-2.0', '31-0.0', '21003-2.0', '24419-2.0']":'green', "['site5', 'site6', '26521-2.0', '31-0.0', '21003-2.0', '24419-2.0', '25741-2.0']":'blue'}
            ax.scatter(x, nlogp, c=subset.covars.map(colors))#, c=z, s=.5)
            plt.xlabel(group+ "IDPs vs"+affect+"grouped by covar adjustment")
            plt.ylabel("-log10p")
            plt.show()
        except:
            pass
