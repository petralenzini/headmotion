import numpy as np
import nibabel as nib
import pandas as pd
import scipy
import h5py
import hdf5storage
import seaborn as sns
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import statistics

def save_image(filename, data, reference_image, *, mask_inds=None):#, dtype=None):
    if len(data.shape) == 1:
        image_shape = reference_image.shape
    elif len(data.shape) == 2:
        image_shape = reference_image.shape + (data.shape[1],)
    else:
        raise ValueError("Data must be 1D or 2D.")

    # Reshape to 3D volume
    if mask_inds is not None:
        image_data = np.zeros(image_shape)
        image_data[mask_inds] = data
    # Retype
    dtype = np.float32
    if dtype is not None:
        image_data = image_data.astype(dtype)

    #  Turn into spatial image
    image = type(reference_image)(
        image_data, reference_image.affine,
        reference_image.header, reference_image.extra)
    image.set_data_dtype(image_data.dtype)
    image.header['cal_max'] = 5.0;
    image.header['cal_min'] = -5.0
    # https://brainder.org/2012/09/23/the-nifti-file-format/
    # image.header.set_intent(1001, name="PFM maps")
    filename += ".nii.gz"
    nib.save(image, filename)

    return image_data


def get_mean_inner_product(W1, W2):
    wlen1 = np.sqrt(np.sum(np.power(W1, 2), axis=0))
    wlen2 = np.sqrt(np.sum(np.power(W2, 2), axis=0))
    if np.any(wlen1 == 0):
        wlen1[wlen1 == 0] = 1
    if np.any(wlen2 == 0):
        wlen2[wlen2 == 0] = 1
    W1_1 = np.multiply(W1, np.divide(1, wlen1))
    W2_1 = np.multiply(W2, np.divide(1, wlen2))

    inner_product = np.matmul(np.transpose(W1_1), W2_1)
    dist = 2 * (1 - inner_product)
    matching_row, matching_col = scipy.optimize.linear_sum_assignment(dist)

    inner_product_noscale = np.matmul(np.transpose(W1), W2)
    #cosinesim = np.multiply(inner_product_noscale, np.divide(1, wlen1 * wlen2))
    cosinesim = np.multiply(inner_product_noscale, np.divide(1, np.outer(wlen1,wlen2)))
    cmatching_row, cmatching_col = scipy.optimize.linear_sum_assignment(1-cosinesim)
    #print("hello",cmatching_row,cmatching_col)
    overlap = np.zeros((wlen1.shape[0],))
    coverlap = np.zeros((wlen1.shape[0],))
    #for i in range(wlen1.shape[0]):
    #    overlap[i] = inner_product[i, matching_col[i]]
    #    coverlap[i] =  cosinesim[i, cmatching_col[i]]

    mean_inner_product = np.mean(overlap)
    return overlap,coverlap,inner_product,cosinesim,matching_col,matching_row,cmatching_col,cmatching_row

def ari(mat1, mat2):
    #identify component with max value for winner takes all cluster assignment
    idx1 = np.argmax(mat1, 0)
    idx2 = np.argmax(mat2, 0)
    return adjusted_rand_score(idx1, idx2)

def winner(mat):
    win=np.zeros([mat.shape[0], mat.shape[1]], dtype=int)
    idx=np.argmax(mat,0)
    for i in range(0,mat.shape[1]):
        win[idx[i], i] = 1
    #print(win.sum(1))
    return win

def matchwides(p_masked,n_masked,outdir,fixcolorder=False,posonly=True,xlab='B',ylab='A'):
    if posonly:
        p_masked[p_masked<0]=0
    rankp=p_masked.shape[0]
    rankn=n_masked.shape[0]
    W1=p_masked.transpose()
    W2=n_masked.transpose()
    #all the extra stuff goes to show that what we were calling an 'adjusted' inner product is actually cosine sim after fixing the denom issue
    overlap,coverlap,inner_product,cosinesim,matching_col,matching_row,cmatching_col,cmatching_row=get_mean_inner_product(W1,W2)
    nomatch = [a for a in list(range(0, rankp)) if a not in matching_row]
    W1reorder = W1.transpose()[list(matching_row)+nomatch].transpose().copy()
    if fixcolorder:
        matching_col=fixcolorder
    nomatchn = [a for a in list(range(0, rankn)) if a not in matching_col]
    W2reorder = W2.transpose()[list(matching_col)+nomatchn].transpose().copy()
    overlap2,coverlap2,inner_product2,cosinesim2,matching_col2,matching_row2,cmatching_col2,cmatching_row2=get_mean_inner_product(W1reorder,W2reorder)
    sns.heatmap(inner_product2, linewidth=0.5, cmap='coolwarm',
                     cbar_kws={'ticks': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1)
    plt.title("Component Cosine Similarity")
    plt.xlabel(xlab)
    plt.ylabel(ylab)  # ,rotation=90)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6, rotation=90)
    plt.axis('equal')
    plt.tight_layout()  # neccessary to get the x-axis labels to fit
    plt.savefig(outdir + "CosineSim_RankN" + str(rankn) + "_RankP" + str(rankp) + '.png')
    plt.show()
    # define the cmap with clipping values
    my_cmap = plt.cm.coolwarm
    my_cmap.set_over("white")
    my_cmap.set_under("white")

    sns.heatmap(inner_product2, linewidth=0.5, cmap=my_cmap,
                cbar_kws={'ticks': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0.5, vmax=1)

    plt.title("Component Cosine Similarity")
    plt.xlabel(xlab)
    plt.ylabel(ylab)#,rotation=90)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6,rotation=90)
    plt.axis('equal')
    plt.tight_layout()  # neccessary to get the x-axis labels to fit
    #plt.savefig(outdir+"HighCorr_CosineSim_RankN"+str(rankn)+"_RankP"+str(rankp) + '.png')
    plt.show()
    return inner_product2, matching_row,matching_col,inner_product

def mask4Matrices(gzs,rank,decomp,mask_inds,n_mask):
#reimport and mask from gz with nib to be certain coords in same space of mask
    p_masked = np.zeros([rank, n_mask], dtype=np.float64)
    for i in range(0, rank):
        #ppath=gzs+decomp+str(i)+"_rank"+str(rank)+".nii.gz"
        ppath=gzs+decomp+"_rank" + str(rank) + "_mode" + str(i)+".nii.gz"
        print(ppath)
        pimage=nib.load(ppath)
        p_masked[i]=pimage.get_fdata()[mask_inds]
    return p_masked #,po_masked

def orientPFM(pfm_hdf5,pathout,rank,decomp,reference_nifti,mask_inds):
    with h5py.File(pfm_hdf5, 'r') as hdf5_file:
        data = hdf5_file['dataset'][...].T  # Deals with column-/row-major ordering
    #save niftis for profumo (i.e. project back to full MNI)
    #p_fullMNI = np.zeros([20, 902629], dtype=np.intp)
    for i in range(0, rank):
        maps = data[:, i]
        name = decomp+"_rank"  + str(rank) +"_mode" + str(i)
        map_directory = pathout
        imagetest=save_image(
            map_directory+name, maps,
            reference_nifti, mask_inds=mask_inds)

def orientNMF(pathin,rank,pathout,reference_nifti):
    clist = [item for item in range(0, rank)]
    Ncolumns = ["n" + str(sub) for sub in clist]
    c = hdf5storage.loadmat(pathin, variable_names=["C"])
    n = hdf5storage.loadmat(pathin, variable_names=["B"])
    eps=0.00001
    # housekeeping
    clist = [item for item in range(0, rank)]
    Ncolumns = ["n" + str(sub) for sub in clist]
    # export and import nii to confirm that we're in the right orientation
    for i in clist:
        n0 = np.reshape(n['B'][:, i], newshape=reference_nifti.shape, order='F')
        nf0 = nib.Nifti1Image(n0, reference_nifti.affine, header=reference_nifti.header)
        nib.save(nf0,
             filename=pathout+"N_rank"  + str(rank) +"_mode" + str(i) +".nii.gz")

def offdiag(i):
    rankp=i.shape[0]
    rankn=i.shape[1]
    offlist=[]
    for n in range(0,rankn):
        for p in range(0,rankp):
            if n !=p:
                #print(n,p)
                offlist.append(i[p,n])
    offsum=sum(offlist)
    offmean=statistics.mean(offlist)
    offmedian=statistics.median(offlist)
    offmin=min(offlist)
    offmax=max(offlist)
    return offmean, offmin, offmedian,offmax

def swarm(vals,outdir,cohort='Profumo'):
    #vals is Dataframe of concatenated contents of a results files
    #cohort is the variable name of stats group you want to keep
    #list of ranks
    #vals.sort_values('rankm', inplace=True)
    subgroup=vals.loc[vals.method==cohort].copy()
    subgroup['rankm']=subgroup.rankm.astype('int')
    #stretch out the figure
    fig,ax = plt.subplots(figsize=(18,5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    #create a swarm plot
    sns.swarmplot(data=subgroup,
                x="rankm",
                y="CosSim",
                size=5,
                #hue="rankm", #trick swarm to plot different colors for groups
                legend=False, #turn this back on if you want to get rid of xlabel altogether
                ax=ax)
    meanpointprops = dict(marker='D', markeredgecolor='black',
                        markerfacecolor='black')
    #adding a boxplot where everything is invisible except the mean
    sns.boxplot(showmeans=True,
              meanline=False,
              meanprops=meanpointprops,
              medianprops={'visible': False},
              whiskerprops={'visible': False},
              zorder=10,
              x="rankm",
              y="CosSim",
              data=subgroup,
              showfliers=False,
              showbox=False,
              showcaps=False)
    plt.xticks(rotation=90,fontsize=14)
    plt.xlabel("")#"ICD Category of Patient Group")
    plt.ylabel("Cosine Similarity",fontsize=14)
    plt.ylim(0, 1)
    plt.title(cohort,fontsize=18) # defaults to Brain rep name
    plt.tight_layout() #neccessary to get the x-axis labels to fit
    #plt.axhline(y=0.5, color='lightgrey', linestyle='-',lw=6)
    plt.axhline(y=0.7, color='lightgrey', linestyle='-',lw=1)
    plt.savefig(outdir+'swarm.png')
    plt.show()

