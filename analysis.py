import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import keras
from sklearn.model_selection import train_test_split


def cal_var_ratio(dFrame):
  temp = dFrame.argmax(axis = -1)
  md, count = scipy.stats.mode(temp)
  vRatio = np.subtract(1, np.divide(count, dFrame.shape[0]))
  return md, vRatio

def cal_pred_entropy(dFrame):
  temp = dFrame.mean(axis = 0)
  t1 = np.log2(temp)
  t1 = np.multiply(t1, temp)
  t1 = np.negative(np.nansum(t1, axis=-1))
  return t1

def cal_mutual_info(dFrame):
  temp = cal_pred_entropy(dFrame)
  t1 = np.log2(dFrame)
  t1 = np.multiply(dFrame, t1)
  t1 = np.nansum(t1, axis = -1)
  t1 = t1.mean(axis=0)
  t1 = np.add(t1, temp)
  return t1

def get_stats(yTrue, mc_ensemble_pred, sts):
  tr = yTrue == mc_ensemble_pred
  fl = yTrue != mc_ensemble_pred
  nm = yTrue == 0
  tlk = yTrue == 1
  ywn = yTrue == 2
  nmP = mc_ensemble_pred == 0
  tlkP = mc_ensemble_pred == 1
  ywnP = mc_ensemble_pred == 2
  yTL = [nm, tlk, ywn]
  mcPL = [nmP, tlkP, ywnP]
  trL = [ sts[tr & x] for x in yTL]
  flL = [[ sts[fl & yTL[i] & mcPL[j]] for j in range(0,3) if i != j] for i in range(0,3)]
  return trL, flL

def createPlot(trL, flL, xlabel, n_bins = 5):
  pltStats = [trL, flL[0], flL[1], flL[2]]
  lbl = [ 'Normal', 'Talking', 'Yawning']
  tlbl = [[ lbl[j] for j in range(0,3) if i != j] for i in range(0,3) ]
  labels = []
  labels.append(lbl)
  labels.extend(tlbl)
  clr = ['red', 'tan', 'lime']
  clrl = [[ clr[j] for j in range(0,3) if i != j] for i in range(0,3) ]
  colours = []
  colours.append(clr)
  colours.extend(clrl)
  title = ['A', 'B', 'C', 'D']
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
  axes = axes.flatten()
  for i in range(0,4):
    axes[i].hist( pltStats[i] , bins = n_bins, label=labels[i], color = colours[i])
    axes[i].legend(prop={'size': 10})
    axes[i].set_xlabel(xlabel)
    axes[i].set_ylabel('Count')
    axes[i].set_title(title[i])
  fig.tight_layout()
  plt.show()
  return fig

def plot_highUnc(stats2, stats3, mcPred, yTrue, mc_ensemble_pred, n_bins = 10):
  psts1 = stats2.argsort()[-6:][::-1]
  psts2 = stats3.argsort()[-6:][::-1]
  psts = np.union1d(psts1, psts2)
  i = 1
  while(psts.shape[0] < 12):
     temp = stats2.argsort()[-(6+i):-(6+i-1)]
     psts = np.union1d(psts, temp)
     i+=1
  lbl = [ 'Normal', 'Talking', 'Yawning']
  clr = ['red', 'tan', 'lime']
  fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 12))
  axes = axes.flatten()
  yt = yTrue[psts]
  pt = mc_ensemble_pred[psts]
  tm = mcPred[:, psts, :]
  for i in range(0,12):
    axes[i].hist( tm[:,i,:] , bins = n_bins, label=lbl, color = clr)
    axes[i].legend(prop={'size': 10})
    axes[i].set_xlabel('Probability')
    axes[i].set_ylabel('Count')
    axes[i].set_title('Predicted: ' + lbl[pt[i]] + ', True: ' + lbl[yt[i]])
  fig.tight_layout()
  plt.show()
  return fig, psts

def plot_incorrect(stats2, mcPred, yTrue, mc_ensemble_pred,x_test, n_bins = 10, leave = 0, datasetType = 'single'):
  msk = yTrue != mc_ensemble_pred
  mskY = yTrue == 2
  stats2req = stats2[msk&mskY]
  mcPredreq = mcPred[:,msk&mskY,:]
  x_testreq = x_test[msk&mskY]
  ytr = yTrue[msk&mskY]
  mctr = mc_ensemble_pred[msk&mskY]
  args = stats2req.argsort()[leave: leave+6]
  lbl = [ 'Normal', 'Talking', 'Yawning']
  clr = ['red', 'tan', 'lime']
  if datasetType == 'single':
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 12))
    # axes = axes.flatten()
    for i in range(0,3):
      axes[0,i].imshow(x_testreq[args[i]])
      axes[0,i].set_title('Predicted: ' + lbl[mctr[args[i]]] + ', True: ' + lbl[ytr[args[i]]])
      axes[1,i].hist( mcPredreq[:,args[i],:] , bins = n_bins, label=lbl, color = clr)
      axes[1,i].legend(prop={'size': 10})
      axes[1,i].set_xlabel('Probability')
      axes[1,i].set_ylabel('Count')
    for i in range(3,6):
      axes[2,i-3].imshow(x_testreq[args[i]])
      axes[2,i-3].set_title('Predicted: ' + lbl[mctr[args[i]]] + ', True: ' + lbl[ytr[args[i]]])
      axes[3,i-3].hist( mcPredreq[:,args[i],:] , bins = n_bins, label=lbl, color = clr)
      axes[3,i-3].legend(prop={'size': 10})
      axes[3,i-3].set_xlabel('Probability')
      axes[3,i-3].set_ylabel('Count')
  elif datasetType == 'multi':
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(14, 12))
    for i in range(0,6):
      axes[i, 0].imshow(x_testreq[args[i]])
      axes[i, 0].set_title('Predicted: ' + lbl[mctr[args[i]]] + ', True: ' + lbl[ytr[args[i]]])
      axes[i, 1].hist( mcPredreq[:,args[i],:] , bins = n_bins, label=lbl, color = clr)
      axes[i, 1].legend(prop={'size': 10})
      axes[i, 1].set_xlabel('Probability')
      axes[i, 1].set_ylabel('Count')		    
  fig.tight_layout()
  plt.show()
  return fig

def plot_correct(stats2, mcPred, yTrue, mc_ensemble_pred,x_test, n_bins = 10, leave = 0, datasetType = 'single'):
  msk = yTrue == mc_ensemble_pred
  mskY = yTrue == 2
  stats2req = stats2[msk&mskY]
  mcPredreq = mcPred[:,msk&mskY,:]
  x_testreq = x_test[msk&mskY]
  ytr = yTrue[msk&mskY]
  mctr = mc_ensemble_pred[msk&mskY]
  args = stats2req.argsort()[leave: leave+6]
  lbl = [ 'Normal', 'Talking', 'Yawning']
  clr = ['red', 'tan', 'lime']
  if datasetType == 'single':
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 12))
    # axes = axes.flatten()
    for i in range(0,3):
      axes[0,i].imshow(x_testreq[args[i]])
      axes[0,i].set_title('Predicted: ' + lbl[mctr[args[i]]] + ', True: ' + lbl[ytr[args[i]]])
      axes[1,i].hist( mcPredreq[:,args[i],:] , bins = n_bins, label=lbl, color = clr)
      axes[1,i].legend(prop={'size': 10})
      axes[1,i].set_xlabel('Probability')
      axes[1,i].set_ylabel('Count')
    for i in range(3,6):
      axes[2,i-3].imshow(x_testreq[args[i]])
      axes[2,i-3].set_title('Predicted: ' + lbl[mctr[args[i]]] + ', True: ' + lbl[ytr[args[i]]])
      axes[3,i-3].hist( mcPredreq[:,args[i],:] , bins = n_bins, label=lbl, color = clr)
      axes[3,i-3].legend(prop={'size': 10})
      axes[3,i-3].set_xlabel('Probability')
      axes[3,i-3].set_ylabel('Count')
  elif datasetType == 'multi':
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(14, 12))
    for i in range(0,6):
      axes[i, 0].imshow(x_testreq[args[i]])
      axes[i, 0].set_title('Predicted: ' + lbl[mctr[args[i]]] + ', True: ' + lbl[ytr[args[i]]])
      axes[i, 1].hist( mcPredreq[:,args[i],:] , bins = n_bins, label=lbl, color = clr)
      axes[i, 1].legend(prop={'size': 10})
      axes[i, 1].set_xlabel('Probability')
      axes[i, 1].set_ylabel('Count')		    
  fig.tight_layout()
  plt.show()
  return fig