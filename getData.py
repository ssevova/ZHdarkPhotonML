#!/usr/bin/env python
"""
Script to get data and write to dataframes for ML training/testing
"""
__author__ = "Stanislava Sevova, Elyssa Hofgard"
###############################################################################                                   
# Import libraries                                                                                                
################## 
import argparse
import sys
import os
import re
import glob
import shutil
import uproot as up
import uproot_methods
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
###############################################################################                                   
# Command line arguments
######################## 
def getArgumentParser():
    """ Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Script to get data and write to dataframes for ML training/testing")
    parser.add_argument('-i',
                        '--indir',
                        dest='indir',
                        help='Directory with input files',
                        default="/afs/cern.ch/work/s/ssevova/public/dark-photon-atlas/plotting/trees/v08/tight-and-ph-skim/mc16d/")
    parser.add_argument('-o',
                        '--output',
                        dest='outdir',
                        help='Output directory for plots, selection lists, etc',
                        default='outdir')
    
    return parser
###############################################################################                                   
# Dataframes for each sample
############################ 
def sampleDataframe(infiles,treename): 
    """ Open the ROOT file(s) corresponding to the treename
    and put the relevant branches into a dataframe """
    
    ## Branches to read in 
    branches = ["w","in_vy_overlap",
                "trigger_lep",
                "passJetCleanTight",
                "n_mu","n_el","n_ph","n_bjet","n_baseph",
                "ph_pt","ph_eta","ph_phi","ph_isGood",
                "met_tight_tst_et", "met_tight_tst_phi",
                "mu_pt","mu_eta", "mu_phi",
                "el_pt","el_eta", "el_phi",
                "metsig_tst",
                "eventNumber",
                "passVjetsFilterTauEl"]
    df_sample = []
    sam_ignore = ["700013","700020","700024","363724"]

    tree = None
    for path, file, start, stop, entry in up.iterate(
            infiles+treename+"*.root",
            treename,
            branches=branches,
            reportpath=True, reportfile=True, reportentries=True):
        
        if any(x in str(path) for x in sam_ignore):
            print('==> IGNORING sample: %s ...'%path)
            continue
        else: 
            print('==> Processing sample: %s ...'%path)
            tree = up.open(path)[treename]

        if tree is not None:
            df_sample.append(tree.pandas.df(branches,flatten=False))
    df_sample = pd.concat(df_sample)
    return df_sample

def calcmT(met,ph):
    return np.sqrt(2*met.pt*ph.pt*(1-np.cos(ph.delta_phi(met))))

def calcPhiLepMet(lep1,lep2,met,ph):
    return np.abs((lep1+lep2).delta_phi(met+ph))
        
def calcAbsPt(lep1,lep2,met,ph):
    pt_lep = (lep1+lep2).pt
    pt_ph_met = (met+ph).pt    
    return np.abs(pt_ph_met-pt_lep)/pt_lep

def getLorentzVec(df,lepType):
    """ Calculates Lorentz vectors for the leptons and photons for bkg/sig:
    but first converts all pTs and masses from MeV to GeV"""
    # Converting weights to true yield
    df['w'] = df['w'] * 36000
    #df['metsig_tst'] = df['metsig_tst'].truediv(np.sqrt(1000))
    df['mu_pt']   = df['mu_pt'].truediv(1000)
    df['el_pt']   = df['el_pt'].truediv(1000)
    df['mu_mass'] = df['mu_mass'].truediv(1000)
    df['el_mass'] = df['el_mass'].truediv(1000)
    df['ph_pt']   = df['ph_pt'].truediv(1000)
    df['met_tight_tst_et'] = df['met_tight_tst_et'].truediv(1000)
    df['ph_pt'] = np.concatenate(df.ph_pt.values).ravel().tolist()

    if lepType == 'n_mu':
        lep_pt   = np.asarray(df.mu_pt.values.tolist()) 
        lep_eta  = np.asarray(df.mu_eta.values.tolist())
        lep_phi  = np.asarray(df.mu_phi.values.tolist())
        lep_mass = np.asarray(df.mu_mass.values.tolist())
    else:
        lep_pt   = np.asarray(df.el_pt.values.tolist()) 
        lep_eta  = np.asarray(df.el_eta.values.tolist())
        lep_phi  = np.asarray(df.el_phi.values.tolist())
        lep_mass = np.asarray(df.el_mass.values.tolist())
            
    lep1 = uproot_methods.TLorentzVectorArray.from_ptetaphim(lep_pt[:,0],lep_eta[:,0],lep_phi[:,0],lep_mass[:,0])
    lep2 = uproot_methods.TLorentzVectorArray.from_ptetaphim(lep_pt[:,1],lep_eta[:,1],lep_phi[:,1],lep_mass[:,1])
    
    ph = uproot_methods.TLorentzVectorArray.from_ptetaphim(df['ph_pt'].to_numpy().astype(float),
                                                           df['ph_eta'].to_numpy().astype(float),
                                                           df['ph_phi'].to_numpy().astype(float),
                                                           0.00)
    met = uproot_methods.TLorentzVectorArray.from_ptetaphim(df['met_tight_tst_et'].to_numpy(),
                                                            0.00, df['met_tight_tst_phi'].to_numpy(), 0.00)
    return lep1,lep2,ph,met

def calcVars(df):
    
    lepType = ['n_mu','n_el']

    df_list = []
    for lep in lepType:
        df_lep = df[df[lep]==2]

        vLep1,vLep2,vPh,vMET = getLorentzVec(df_lep, lep)

        df_lep['mT'] = calcmT(vMET,vPh)
        df_lep['dphi_mety_ll'] = calcPhiLepMet(vLep1,vLep2,vMET,vPh)
        df_lep['AbsPt'] = calcAbsPt(vLep1,vLep2,vMET,vPh)
        df_lep['Ptll'] = (vLep1 + vLep2).pt
        df_lep['Ptllg'] = (vLep1 + vLep2 + vPh).pt
        df_lep['mll'] = (vLep1 + vLep2).mass
        df_lep['mllg'] = (vLep1+vLep2+vPh).mass
        df_lep['lep1pt'] = vLep1.pt
        df_lep['lep2pt'] = vLep2.pt
        df_lep['dphi_met_ph'] = np.abs(vMET.delta_phi(vPh))
    
        df_list.append(df_lep)
    
    new_df = pd.concat(df_list)

    return new_df
 
def main(): 
    """ Run script"""
    options = getArgumentParser().parse_args()

    ### Make output dir
    dir_path = os.getcwd()
    out_dir = options.outdir
    path = os.path.join(dir_path, out_dir)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.chdir(path)

    ### Make all the bkg and signal dataframes
    ## Z+jets
    df_zjets  = sampleDataframe(options.indir,"Z_strongNominal")
    df_zjets  = df_zjets.append(sampleDataframe(options.indir,"Z_EWKNominal"))
    ## Z+photon
    df_zgamma = sampleDataframe(options.indir,"Zg_strongNominal")

    ## ttbar/single top/Wt/ttbar+V
    df_top    = sampleDataframe(options.indir,"ttbarNominal")
    df_top    = df_top.append(sampleDataframe(options.indir,"ttVNominal"))

    ## Triboson
    df_VVV    = sampleDataframe(options.indir,"VVVNominal")
    df_VVV    = df_VVV.append(sampleDataframe(options.indir,"VVyNominal"))

    ## Diboson
    df_VV     = sampleDataframe(options.indir,"VVNominal")
    df_VV     = df_VV.append(sampleDataframe(options.indir,"VV_ewkNominal"))
    df_VV     = df_VV.append(sampleDataframe(options.indir,"ggZZNominal"))
    df_VV     = df_VV.append(sampleDataframe(options.indir,"ggWWNominal"))

    ## H->Zy
    df_HZy    = sampleDataframe(options.indir,"ggH125ZyNominal")
    df_HZy    = df_HZy.append(sampleDataframe(options.indir,"ttH125ZyNominal"))
    df_HZy    = df_HZy.append(sampleDataframe(options.indir,"VBFH125ZyNominal"))
    df_HZy    = df_HZy.append(sampleDataframe(options.indir,"VH125ZyNominal"))

    ## signal
    df_sig = sampleDataframe(options.indir,"HyGrNominal")
    

    ## Remove overlapping Z+jets events
    df_zjets = df_zjets[df_zjets['in_vy_overlap'] > 0]
    ## Make collective bkg dataframe
    df_bkg = df_zjets
    #df_bkg.reset_index(inplace=True)
    df_bkg = df_bkg.append(df_zgamma).append(df_top).append(df_VVV).append(df_VV).append(df_HZy)
    df_bkg['event'] = list(np.zeros(len(df_bkg)))
    df_sig['event'] = list(np.full(len(df_sig),1))
    print(df_bkg)
    print(df_bkg[df_bkg.index.duplicated()]) 
    print(df_bkg.columns.duplicated())
    df_bkg = df_bkg[(df_bkg['n_mu']==2) | (df_bkg['n_el']==2)]
    df_bkg = df_bkg[(df_bkg['trigger_lep']>0) &
                    (df_bkg['passJetCleanTight']==1) &
                    (df_bkg['n_ph']==1) &
                    (df_bkg['n_baseph']==1) &
                    (df_bkg['n_bjet']==0) &
                    (df_bkg['passVjetsFilterTauEl']==True)]
    df_bkg['mu_mass'] = list(np.full((len(df_bkg),2),105.6))
    df_bkg['el_mass'] = list(np.full((len(df_bkg),2),0.511))

    df_sig = df_sig[(df_sig['n_mu']==2) | (df_sig['n_el']==2)]
    df_sig = df_sig[(df_sig['trigger_lep']>0) &
                    (df_sig['passJetCleanTight']==1) &
                    (df_sig['n_ph']==1) &
                    (df_sig['n_baseph']==1) &
                    (df_sig['n_bjet']==0) &
                    (df_sig['passVjetsFilterTauEl']==True)]

    df_sig['mu_mass'] = list(np.full((len(df_sig),2),105.6))
    df_sig['el_mass'] = list(np.full((len(df_sig),2),0.511))

    df_bkg = calcVars(df_bkg)
    df_bkg = df_bkg[(df_bkg['mll'] > 66) &
                    (df_bkg['mll'] < 116) &
                    (df_bkg['lep1pt'] > 26) &
                    (df_bkg['lep2pt'] > 7) &
                    (df_bkg['ph_pt'] > 25) &
                    (df_bkg['met_tight_tst_et']> 50)]# & 

    df_sig = calcVars(df_sig)
    df_sig = df_sig[(df_sig['mll'] > 66) &
                    (df_sig['mll'] < 116) &
                    (df_sig['lep1pt'] > 26) &
                    (df_sig['lep2pt'] > 7) &
                    (df_sig['ph_pt'] > 25) &
                    (df_sig['met_tight_tst_et'] > 50)]
    all_data = pd.concat([df_bkg,df_sig])
    # write out dataframes
    df_bkg.to_csv("bkg_data")
    df_sig.to_csv("sig_data")
    all_data.to_csv("all_data")    

if __name__ == '__main__':
    main()
