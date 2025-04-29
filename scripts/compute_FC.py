
# NEED TO HAVE RUN:
#   F02_pipeline_BIDS
#   F02_pipeline_denoising
#   F02_pipeline_ROIs
#   F02_pipeline_cerebellum


# Computes measures of connectivity:
#   fALFF
#   ReHo_Kendall
#   RSFA
#   ReHo-Cohe - REST
#   Correlation
#   Correlation with Shrinkage
#   Euclidean distance
#   Mannhathan distance
#   Wasserstein distance
#   Dynamic Time Warping
#   Cross-correlation
#   Partial correlation
#   Magnitude-square Coherence
#   Wavelet-Coherence
#   Mutual information

import numpy as np, pandas as pd, os, glob, itertools, errno, re, csv, shutil
import multiprocessing as mp

from nipype.interfaces import fsl,spm,afni,utility
from nipype import Workflow,Node,Function,JoinNode,MapNode,IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.confounds import ACompCor
from nipype.interfaces.base import TraitedSpec,CommandLineInputSpec,CommandLine, \
    File,OutputMultiPath,traits, Directory, BaseInterface, BaseInterfaceInputSpec, isdefined
from nipype.utils.filemanip import split_filename
from nipype.algorithms.misc import Gunzip
from pathlib2 import Path
from string import Template
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
import json
from subprocess import call
from misc_funs import exclude_subs

AG_EXP = {'A02':['Extinction_EEG_fMRI'], 'A03':['3T','3T_EEG'], 'A05':['study3'],
          'A08':['ATOACQ','ContextII','Cort','TIA'], 
          'A09':['Extinction_Generalization_I','Extinction_Generalization_II'],
          'A12':['study_1','study_2']}

trim_values = [2,None]
desclab = 'trimmed{}'.format(''.join(map(str,trim_values)))
space = 'T1w'
confounds = 'Satterthwaite'
which_RS = '*task-rest*_space-{space}*conftype-{conf}_desc-{desc}_bold.nii.gz'.format(space=space,
                                                                                      conf=confounds,
                                                                                      desc=desclab)
descatlas = 'Default'
which_atlas = '*space-{space}*desc-{datlas}_dseg*'.format(space=space,
                                                          datlas=descatlas)
overwrite = True

for AG, studies in AG_EXP.items():
    for exp in studies:
        
        # Specify directories and participant list
        base_dir = '/media/f02/F02'
        bids_dir = os.path.join(base_dir, AG, exp)
        wf_dir = os.path.join(bids_dir, 'workflows')
        rawdata_dir = os.path.join(bids_dir, 'rawdata')
        derivatives_dir = os.path.join(bids_dir, 'derivatives')
        func_denoised_dir = os.path.join(derivatives_dir, 'func_denoised')
        ROIs_dir = os.path.join(derivatives_dir, 'ROIs')
        fmriprep_dir = os.path.join(derivatives_dir, 'fmriprep')
        freesurfer_dir = os.path.join(derivatives_dir, 'freesurfer')
        crashdir = os.path.join(bids_dir, 'crashfiles')
        FC_ROI_dir = os.path.join(derivatives_dir, 'FC_ROI')
        
        
        participants_file = pd.read_csv(os.path.join(rawdata_dir, 'participants.tsv'),
                                        sep='\t', usecols=['participant_id','rsfMRI','T1w'])
        # select only subjects that have both T1w and resting-state data
        vptcs = participants_file[['rsfMRI','T1w']].prod(1).astype(bool)
        participants = participants_file.loc[vptcs,'participant_id'].tolist()
        
        mp_jobs = mp.cpu_count()
        
        ###############################################################################
        ######################## SELECT SUBJECTS/FILES ################################
        ###############################################################################
        
        fsl.FSLCommand.set_default_output_type('NIFTI')
        
        # =============================================================================
        # INFOSOURCE - A FUNCTION FREE NODE TO ITERATE OVER THE LIST OF SUBJECT NAMES
        # =============================================================================
        infosource = Node(IdentityInterface(fields=['ptc']), name="infosource")
        infosource.iterables = [('ptc', participants)]
        
        sub_wf_dir = os.path.join(wf_dir, 'wf_FC_ROI', space)
        if not os.path.exists(sub_wf_dir): 
            os.makedirs(sub_wf_dir)
            
        name_wf = descatlas + '_' + desclab
        
        FC_ROI_group_dir = os.path.join(FC_ROI_dir, 'group', space, name_wf)
        if not os.path.exists(FC_ROI_group_dir): 
            os.makedirs(FC_ROI_group_dir)
        
        
        wf_FC_ROI = Workflow(name=name_wf, base_dir=sub_wf_dir)
        wf_FC_ROI.config['execution'] =  {'stop_on_first_crash': False,
                                          'hash_method': 'content',
                                          'use_relative_paths': False,
                                          #'keep_inputs': True, 
                                          'remove_unnecessary_outputs': False,
                                          'crashfile_format': 'txt',
                                          'crashdump_dir': crashdir}
        
        templates = {'bold_data': os.path.join(fmriprep_dir,'{ptc}', 'func','*task-rest*space-{space}*desc-preproc_bold.nii.gz'.format(space=space)),
                     'denoised_func': os.path.join(func_denoised_dir,'{ptc}', '**','{}'.format(which_RS)),
                     'bold_mask': os.path.join(fmriprep_dir,'{ptc}', 'func','*task-rest*space-{space}*desc-brain_mask.nii.gz'.format(space=space)),
                     'atlas': os.path.join(ROIs_dir,'{ptc}','func', '{}.nii.gz'.format(which_atlas)),
                     'atlas_labels': os.path.join(ROIs_dir,'{ptc}', 'func', '{}.json'.format(which_atlas)),
                     'func_json': os.path.join(rawdata_dir,'{ptc}', 'func','*task-rest*_bold.json')}
        
        select_files = Node(SelectFiles(templates), name='select_files')
         
        # =============================================================================
        # select valid rois for 3dRSFC - exclude only-zero ROIs
        # =============================================================================
        
        def get_info(func_file, func_json, add_info={}):
            import os, json
            with open(func_json, 'r+') as json_file:
                data = json.load(json_file) 
            TR = data['RepetitionTime']
            f = os.path.basename(func_file)
            l = [pair.split('-') for pair in f.split('_') if len(pair.split('-'))==2]
            d = {k[0]:k[1] for k in l}
            d.update({'TR':TR, **add_info})
            return  d
        
        func_info = Node(Function(function=get_info, input_names=['func_file','func_json','add_info'], 
                                  output_names=['d']), name='func_info')
        func_info.inputs.add_info = {'AG': AG, 'exp':exp, 'task':'rest', 'descatlas':descatlas, 'name_wf':name_wf}
        # =============================================================================
        # UNCOMPRESS WITH GUNZIP - NIPYPE TOOL (SPM-BASED TOOLS  REQUIRE UNCOMPRESSED FILES)
        # =============================================================================
        def gunzip_rois_fun(valid_rois):
            import os,gzip,shutil
            from nipype.utils.filemanip import split_filename
            
            rois_nii=[]
            gunz_rois = os.path.abspath(os.path.curdir)
            for roi in valid_rois:
                _,filename,_ = split_filename(roi)
                with gzip.open(roi, 'rb') as f_in:
                    outfile = os.path.join(gunz_rois,filename+'.nii')
                    with open(outfile, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        rois_nii.append(outfile) 
            return rois_nii
        
        gunzip_rois = Node(Function(function=gunzip_rois_fun, input_names=['valid_rois'],
                                    output_names=['rois_nii']), name='gunzip_rois')
        
        
        # =============================================================================
        # 3dRSFC + REHO KENDALL - AFNI
        # =============================================================================
        
        class RSFCInputSpec(CommandLineInputSpec):
            in_file = File(desc="T1", exists=True, mandatory=True,
                           argstr="-input %s")
            mask_file = File(desc="-mask", exists=True, mandatory=False,
                             argstr="-mask %s")
            orts = File(desc="-ort", exists=True, mandatory=False,
                        argstr="-ort %s")
            bp = traits.Tuple(traits.Float, traits.Float,
                              desc="""Remove""", argstr='-band %g %g')
            prefix = File(desc='name of output skull stripped image',
                          argstr='-prefix %s')
            no_detrend = traits.Bool(argstr='-nodetrend', desc="register to mean volume")
            no_LFF = traits.Bool(argstr='-no_rs_out', desc="register to mean volume")
            func_info = traits.Dict({}, usedefault=True)
            
        class RSFCOutputSpec(TraitedSpec):
            LFF = OutputMultiPath(desc = "T1 file", exists = True)
            metrics = OutputMultiPath(File(desc = "T1 file", exists = True))
            
        class RSFCTask(CommandLine):
            input_spec = RSFCInputSpec
            output_spec = RSFCOutputSpec
            _cmd = '3dRSFC'
            
            def change_filename(self, file):
                from nipype.utils.filemanip import split_filename
                _,filename,ext = split_filename(file)
                method = re.findall('_(.*?)\+',filename,)[0]
                filename = 'sub-{ptc}_space-{space}_bandpass-{bp}_confounds-{conf}_method-afni{method}_map{ext}'.format(ptc=self.inputs.func_info['sub'], 
                                                                                                                        space=self.inputs.func_info['space'],
                                                                                                                        bp=self.inputs.func_info['bandpass'],
                                                                                                                        conf=self.inputs.func_info['confounds'],
                                                                                                                        method=method,
                                                                                                                        ext=ext)
                os.rename(file, os.path.abspath(filename))
                return os.path.abspath(filename)
                                                                                    
            def _list_outputs(self):
                import os
                
                outputs = self.output_spec().get()
                outdir = os.getcwd()
        
                if not self.inputs.no_LFF:
                    outputs["LFF"] = glob.glob(os.path.join(outdir, '*_LFF*'))
                    
                outputs["metrics"] = [os.path.join(outdir,f) for f in os.listdir(outdir) if re.search(r'(ALFF|RSFA).*', f)]
                return outputs
            
        RSFC_derivatives = Node(RSFCTask(no_LFF=False, no_detrend=True, prefix='afni'), name='RSFC_derivatives')
        RSFC_derivatives.inputs.bp = (0.005, 0.1)
        
        
        # AFNI REHO KENDALL
        reho_afni_whole_brain = Node(afni.ReHo(chi_sq=True, out_file='afni_ReHoKCC.nii.gz'), 
                                     name='reho_afni_whole_brain')
        
        def afni_prefix_add(file):
            from nipype.utils.filemanip import split_filename
            _,prefix,ext = split_filename(file)
            return prefix+'_method-afniReHoKCC_map'+ext
        
        afni_prefix = Node(afni.ReHo(chi_sq=True, out_file='afni_ReHoKCC.nii.gz'), name='reho_afni_whole_brain')
        
        
        def get_time_series(atlas,atlas_labels,func_file,ptc,AG,exp):
            import json,os
            import nilearn as nil, nibabel as nib, pandas as pd
            import itertools as it
            from nilearn.input_data import NiftiLabelsMasker
            from misc_funs import extractBetween
             
            masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
            try: 
                ts = masker.fit_transform(func_file)
            except ValueError:
                func_im, atlas_im = nib.load(func_file), nib.load(atlas)
                rfunc_im = nil.image.resample_img(func_im, atlas_im.affine)
                ts = masker.fit_transform(rfunc_im)
             
            with open(atlas_labels) as labels_f:
                labels = json.load(labels_f)
                labels = dict(sorted(labels.items(), key=lambda item: item[1]))
        
            pair_names = list(it.permutations(labels.keys(),2))
            
            df_ts = pd.DataFrame(ts, columns=labels.keys())    
            df_ts[['participant','AG','study']] = ptc,AG,exp
            df_ts.reset_index().rename({'index':'vol'}, inplace=True)
            
            [sp,cf,desc] = extractBetween(os.path.basename(func_file), ['space-','_'],['conftype-','_'],['desc-','_'])
            filename = os.path.abspath('{sub}_space-{sp}_conftype-{cf}_desc-ts_df.tsv'.format(sub=ptc,
                                                                                              sp=sp,
                                                                                              cf=cf))
            df_ts.to_csv(filename, sep='\t', index=False)  
            
            return filename, labels
        
        get_ts = Node(Function(function=get_time_series, 
                               input_names=['atlas','atlas_labels','func_file','ptc','AG','exp'],
                               output_names=['df_ts','labels']), name='get_ts')
        get_ts.inputs.AG = AG
        get_ts.inputs.exp = exp
        
        class DistanceMetricsInputSpec(BaseInterfaceInputSpec):
            ptc = traits.Str()
            df_ts = File()
            kind = traits.Str()
            func_info = traits.Dict({}, usedefault=True)
            args = traits.Dict({}, usedefault=True)
        
        class DistanceMetricsOutputSpec(TraitedSpec):
            df_dist = File(exists=True)
        
        class DistanceMetrics(BaseInterface):
            input_spec = DistanceMetricsInputSpec
            output_spec = DistanceMetricsOutputSpec
        
            def get_vars(self):
                return ([], [])
            
            def _run_interface(self, runtime):
                import numpy as np, multiprocessing as mp
                from misc_funs import flatten
                from itertools import chain
                        
                ts = pd.read_csv(self.inputs.df_ts, sep='\t')
                ts = ts.drop(['AG','study','participant'], axis=1)
        
                pool = mp.Pool(mp_jobs)
                r = pool.map( self.dist_fun, range(ts.shape[1]))
                pool.close()
        
                res,pairs = list(zip(*r))
                
                conn_mat = list(chain.from_iterable(res))
                
                pairs = list(chain.from_iterable(pairs))        
                
                conn_mat_df = pd.DataFrame(pairs, columns=['seed','target'])
                conn_mat_df[['hemi_seed','roi_seed']] = conn_mat_df['seed'].str.split('_',expand=True)
                conn_mat_df[['hemi_target','roi_target']] = conn_mat_df['target'].str.split('_',expand=True)
                
                conn_mat_df[self.inputs.kind] = conn_mat
                conn_mat_df.insert(0, 'participant', 'sub-'+self.inputs.func_info['sub'])
                 
                
                self.filename = 'sub-{ptc}_space-{space}_conftype-{conf}_method-{method}_df.tsv'.format(ptc=self.inputs.func_info['sub'], 
                                                                                                        space=self.inputs.func_info['space'],
                                                                                                        conf=self.inputs.func_info['conftype'],
                                                                                                        method=self.inputs.kind)
                conn_mat_df.to_csv(os.path.abspath(self.filename), sep='\t', index=False)  
                
                return runtime
              
            def dist_fun(self,nx):
                import numpy as np
                
                ts = pd.read_csv(self.inputs.df_ts, sep='\t')
                ts = ts.drop(['AG','study','participant'], axis=1)
                args = self.inputs.args
                roi1 = ts.iloc[:,nx].name
                x = ts.loc[:,roi1]
                res,idx = self.get_vars()
                TR = self.inputs.func_info['TR']
                lcut = 0 if 'lcut' not in args else args['lcut']
        
                for roi2 in ts.columns:
                        # if roi1!=roi2:
                            y = ts.loc[:,roi2]
                            ###### CROSS-CORRELATION ######
                            if self.inputs.kind=='xcorr':
                                norm_x = np.linalg.norm(x)
                                xx = x / norm_x
                                norm_y = np.linalg.norm(y)
                                yy = y / norm_y
                                r = np.correlate(xx,yy,'full')
                                try:
                                    args['lags']
                                except KeyError:
                                    if lcut==0:
                                        lags = [0, len(r)]
                                    else:
                                        lag = int((1/lcut)/TR)
                                        lags = [(len(r)//2)-lag, (len(r)//2)+lag]
                                xcorr = np.max(r[range(*lags)])
                                res.append(xcorr)
                                
                            ###### EUCLIDEAN DISTANCE ######
                            if self.inputs.kind=='EuclideanDist':
                                ED = (np.linalg.norm(x-y)) / np.sqrt(len(x))
                                res.append(ED)
                                
                            ###### MANHATTAN DISTANCE ######
                            if self.inputs.kind=='ManhattanDist':
                                MD = (np.linalg.norm(x-y, ord=1)) / len(x)
                                res.append(MD)
                                
                            ###### WASSERSTEIN DISTANCE ######
                            if self.inputs.kind=='WassersteinDist':
                                from scipy.stats import wasserstein_distance
                                WD = wasserstein_distance(x,y)
                                res.append(WD)
                            
                            ###### DYNAMIC TIME WARPING ######            
                            if self.inputs.kind=='dtw':
                                from dtw import dtw
                                low_cutoff = 1 if lcut==0 else lcut
                                try:
                                    args['window_args']['window_size']
                                except KeyError:
                                    args.update({'window_args':{'window_size':1/low_cutoff}})
                                alignement = dtw(x, y, **args)
                                dtw_TS = alignement.normalizedDistance
                                res.append(dtw_TS)
                                
                            #### LONGEST COMMON SUBSEQUENCE ######    
                            if self.inputs.kind=='lcss':
                                from tslearn import metrics
                                _,lcss = metrics.lcss_path_from_metric(x,y,**args)
                                res.append(np.sqrt(lcss))
                                
                            #### COHERENCE ######                            
                            if self.inputs.kind=='mscohe':    
                                from scipy.signal import coherence
                                import copy
                                
                                args2 = copy.deepcopy(args)
                                args2.update({'window': np.hamming}) if 'window' not in args2 else None
                                args2.update({'nb_samples': len(x)//4.5}) if 'nb_samples' not in args2 else None
                                args2.update({'window': args2['window'](args2['nb_samples'])})
                                args2.update({'noverlap': 0.5*args2['nb_samples']}) if 'noverlap' not in args2 else None
                                args2.update({'fs': TR}) if 'fs' not in args2 else None
                                
                                args2 = {k: args2[k] for k in args2.keys() - {'nb_samples'}}
                                
                                mscohere = np.max( coherence(x,y,**args2)[1] )
                                res.append(mscohere)
                                                    
                            #### WAVELET-COHERENCE ######                            
                            if self.inputs.kind=='wavcohe':    
                                import pycwt
                                wavcohe = pycwt.wct(x,y,**args)
                                res.append(wavcohe)
                            
                            #### MUTUAL INFORMATION ###### 
                            if self.inputs.kind=='MI':
                                from sklearn.feature_selection import mutual_info_regression
                                
                                def mi_rand(x,y,i):
                                    mi = mutual_info_regression(x.values.reshape(-1,1), y, 
                                                                discrete_features=[False], 
                                                                random_state=i)
                                    return mi[0]
                                
                                mi = np.array([mi_rand(x,y,i) for i in range(1000)]).mean()
                                res.append(mi)
                                
                            idx.append((roi1,roi2))
                            
                return res,idx
                    
            def _list_outputs(self):
                outputs = self._outputs().get()
                outputs["df_dist"] = os.path.abspath(self.filename)
                
                return outputs
               
        xcorr = Node(DistanceMetrics(kind='xcorr'), name='xcorr') 
        xcorr.inputs.args = {}
        xcorr.overwrite = overwrite
        
        ED = Node(DistanceMetrics(kind='EuclideanDist'), name='ED') 
        ED.overwrite = overwrite
        
        MD = Node(DistanceMetrics(kind='ManhattanDist'), name='MD') 
        MD.overwrite = overwrite
        
        WD = Node(DistanceMetrics(kind='WassersteinDist'), name='WD') 
        WD.overwrite = overwrite
        
        MI = Node(DistanceMetrics(kind='MI'), name='MI')
        MI.overwrite = overwrite
        
        mscoherence = Node(DistanceMetrics(kind='mscohe'), name='mscoherence') 
        mscoherence.inputs.args = {'window':np.hamming, 'detrend':False, 'nfft':256}
        mscoherence.overwrite = overwrite
        
        wavcoherence = Node(DistanceMetrics(kind='wavcohe'), name='wavcoherence') 
        wavcoherence.inputs.args = {'window':np.hamming, 'detrend':False, 'nfft':256}
        wavcoherence.overwrite = overwrite
        
        DTW = Node(DistanceMetrics(kind='dtw'), name='DTW') 
        DTW.inputs.args = {'window_type':'sakoechiba', 
                           'window_args': {'window_size':100},
                           'step_pattern':'symmetric2', 
                           'dist_method':'sqeuclidean'}
        DTW.overwrite = overwrite
        
        
        LCSS = Node(DistanceMetrics(kind='lcss'), name='LCSS')
        LCSS.inputs.args = {'global_constraint':'sakoe_chiba', 
                            'sakoe_chiba_radius':None, 
                            'metric':'sqeuclidean'}
        LCSS.overwrite = overwrite
        
        class ConnMatrixInputSpec(BaseInterfaceInputSpec):
            ptc = traits.Str()
            df_ts = File()
            estimator = traits.Any()
            kind = traits.Str()
            standardise = traits.Bool(default_value=True, usedefault=True)
            method = traits.Str(default_value='corr', usedefault=True)
            func_info = traits.Dict({}, usedefault=True)
        
        class ConnMatrixOutputSpec(TraitedSpec):
            df_corr = File(exists=True, desc="")
            connmat = File(exists=True, desc="")
        
        
        class ConnMatrix(BaseInterface):
            input_spec = ConnMatrixInputSpec
            output_spec = ConnMatrixOutputSpec
            
            def _run_interface(self, runtime):
                from nilearn.connectome import ConnectivityMeasure
                import pandas as pd
                import os
                
                ts = pd.read_csv(self.inputs.df_ts, sep='\t')
                ts = ts.drop(['AG','study','participant'], axis=1)
                            
                conn_measure = ConnectivityMeasure(cov_estimator=self.inputs.estimator,
                                           kind=self.inputs.kind)
                conn_mat = conn_measure.fit_transform([ts.to_numpy()])[0]
                connmat_file = 'sub-{ptc}_space-{space}_conftype-{conf}_method-{method}_connmat.txt'.format(ptc=self.inputs.func_info['sub'], 
                                                                                                            space=self.inputs.func_info['space'],
                                                                                                            conf=self.inputs.func_info['conftype'],
                                                                                                            method=self.inputs.method)
                np.savetxt(os.path.abspath(connmat_file), conn_mat)
                
                conn_mat_df = pd.DataFrame(conn_mat,columns=ts.columns, index=ts.columns)
                
                df_corr = conn_mat_df.unstack().reset_index().rename(
                                                              columns={'level_0': 'seed',
                                                                       'level_1': 'target',
                                                                       0: self.inputs.method})
                df_corr[['hemi_seed','roi_seed']] = df_corr['seed'].str.split('_',expand=True)
                df_corr[['hemi_target','roi_target']] = df_corr['target'].str.split('_',expand=True)
                
                df_corr.insert(0, 'participant', 'sub-'+self.inputs.func_info['sub'])
                filename = 'sub-{ptc}_space-{space}_conftype-{conf}_method-{method}_df.tsv'.format(ptc=self.inputs.func_info['sub'], 
                                                                                                   space=self.inputs.func_info['space'],
                                                                                                   conf=self.inputs.func_info['conftype'],
                                                                                                   method=self.inputs.method)
        
                df_corr.to_csv(os.path.abspath(filename), sep='\t', index=False)        
                
                setattr(self, '_filename', filename)
                setattr(self, '_conn_mat', connmat_file)            
                return runtime
            
            def _list_outputs(self):
                outputs = self._outputs().get()
                outputs["df_corr"] = os.path.abspath(getattr(self, '_filename'))
                outputs["connmat"] = os.path.abspath(getattr(self, '_conn_mat'))
                return outputs
        
        corr_emp = Node(ConnMatrix(kind='correlation', estimator=EmpiricalCovariance(), method='corrEmpCov'), 
                        name='corr_emp')
        corr_emp.overwrite = overwrite
        corr_LW = Node(ConnMatrix(kind='correlation', estimator=LedoitWolf(), method='corrLW'), 
                       name='corr_LW')
        corr_LW.overwrite = overwrite
        parcorr_LW = Node(ConnMatrix(kind='partial correlation', estimator=LedoitWolf(), method='parCorr'), 
                          name='parcorr_LW')
        parcorr_LW.overwrite = overwrite
        
        
        
        class GraphTheoryInputSpec(BaseInterfaceInputSpec):
            metric = traits.Dict(mandatory=True)
            which_corr = traits.Str('corr', usedefault=True)
            connmat = File(mandatory=True)
            name_rois = traits.Dict()
            out_file = traits.Str()
            func_info = traits.Dict({}, usedefault=True)
        
        class GraphTheoryOutputSpec(TraitedSpec):
            df_mat = File(exists=True, desc="")
        
        class GraphTheory(BaseInterface):
            input_spec = GraphTheoryInputSpec
            output_spec = GraphTheoryOutputSpec
        
            def _run_interface(self, runtime):
                import bct
                import pandas as pd, numpy as np
                from operator import attrgetter 
        
                metric = self.inputs.metric
                connmat = [np.loadtxt(self.inputs.connmat)]

                labels = (self.inputs.name_rois).keys()
                
                df = pd.DataFrame()
                for k,values in metric.items():
                    values if isinstance(values, list) else [values]
                    for v in values:
                        name_metric = '_'.join([k,v])
                        fun = attrgetter('.'.join([k,v]))(bct)
                        r = fun(*connmat)
                        df2 = pd.DataFrame(r, index=labels, columns=[name_metric])
                        df = pd.concat([df, df2], axis=1)  
                df.index.name = 'ROI'
                df.insert(0, 'participant', 'sub-'+self.inputs.func_info['sub'])
        
                k = k+self.inputs.which_corr
                filename = 'sub-{ptc}_space-{space}_conftype-{conf}_method-GT{method}_df.tsv'.format(ptc=self.inputs.func_info['sub'], 
                                                                                                     space=self.inputs.func_info['space'],
                                                                                                     conf=self.inputs.func_info['conftype'],
                                                                                                     method=k)
                df.to_csv(os.path.abspath(filename), sep='\t', index=True)  
        
                setattr(self, '_filename', filename)
        
                return runtime
        
            def _list_outputs(self):
                outputs = self._outputs().get()
                outputs["df_mat"] = os.path.abspath(self._filename)
                return outputs
            
        which_corr = 'corrLW'
        centrality_metrics = Node(GraphTheory(), name='centrality_metrics_{}'.format(which_corr))
        centrality_metrics.inputs.metric = {'centrality':['betweenness_wei',
                                                          'eigenvector_centrality_und']}
        centrality_metrics.inputs.which_corr = which_corr
        centrality_metrics.overwrite = overwrite
        
        
        degree_metrics = Node(GraphTheory(), name='degree_metrics_{}'.format(which_corr)) 
        degree_metrics.inputs.metric = {'degree':['degrees_und',
                                                  'strengths_und']}
        degree_metrics.inputs.which_corr = which_corr
        degree_metrics.overwrite = overwrite
               
        def get_briks(files):
            import re
            from misc_funs import flatten  
            briks = [f for f in list(flatten(files)) if re.search(r'.*BRIK$', f)]
            return briks
        
        def aggregate_afni_conns(atlas,atlas_labels,RSFC,ReHoKCC,func_info,fun='mean'):
            import os, json
            import pandas as pd, numpy as np, nibabel as nib
            from misc_funs import flatten,dict_update
            
            d = {}
            metrics = ['_ALFF','_fALFF','_mALFF','_RSFA','_fRSFA','_mRSFA','_ReHoKCC']
            atlas = nib.load(atlas).get_fdata()
            
            with open(atlas_labels, 'r') as json_file:
                labels = json.load(json_file)
            
            df_fun = pd.DataFrame()
            files = flatten([RSFC,ReHoKCC])
            for f in files:
                metric = next((x[1:] for x in metrics if x in os.path.basename(str(f))), False)
                afnimap = np.squeeze(nib.load(f).get_fdata())
                for lab,lab_nb in labels.items():
                    dat = afnimap[atlas==lab_nb]
                    if metric=='ReHoKCC':
                        l = {lab:{'afniReHoKCCchisq': dat[:,0]}}
                        dict_update(d, l)
                        l = {lab:{'afniReHoKCC': dat[:,1]}}
                        dict_update(d, l)
                        
                    else:
                        l = {lab:{'afni'+metric: dat}}
                        dict_update(d, l)
        
                 
            df_conns = pd.DataFrame.from_dict(d)
            filename = 'sub-{ptc}_space-{space}_conftype-{conf}_values.json'.format(ptc=func_info['sub'], 
                                                                                    space=func_info['space'],
                                                                                    conf=func_info['conftype'])
            
            out_values = os.path.abspath(filename)
            df_conns.to_json(out_values)
            
            dfs = []
            fun = np.mean if fun=='mean' else np.median
            for m in df_conns.index:
                df = df_conns.loc[m,].apply(fun).reset_index().pivot_table(values=m, columns='index').melt(var_name='ROI',value_name=m)
                df[['participant','AG','study']] = func_info['sub'],func_info['AG'],func_info['exp'],
                filename = 'sub-{ptc}_space-{space}_conftype-{conf}_method-{m}_df.tsv'.format(ptc=func_info['sub'], 
                                                                                              space=func_info['space'],
                                                                                              conf=func_info['conftype'],
                                                                                              m=m)  
                outfile = os.path.abspath(filename)
                df.to_csv(outfile, sep='\t')  
                dfs.append(outfile)
            return out_values, dfs
        
        AFNI_conns = Node(Function(function=aggregate_afni_conns,
                                   input_names=['atlas','atlas_labels','RSFC','ReHoKCC','func_info','fun'],
                                   output_names=['out_values','dfs']), name='AFNI_conns')
        
        
        def write_conn_mat(dfs, dfs2, info, group_dir):
            from itertools import chain
            from misc_funs import flatten
            import pandas as pd
            import os, re
            
            all_files = list(flatten([dfs,dfs2]))
            
            pattern = r'method-(.*?)_df'
            matches = [re.findall(pattern,string) for string in all_files]
            methods = set(chain(*matches))
            
            outfiles = []
            for method in methods:
                
                files = [file for file in all_files if 'method-{}_'.format(method) in file]
                dfs_read = (pd.read_csv(f, sep='\t') for f in files)
                df_concat = pd.concat(dfs_read).reset_index(drop=True)
                df_concat.loc[:,['AG','study']] = info['AG'], info['exp']
                
                outfile = os.path.join(group_dir,'space-{space}_conftype-{conf}_method-{method}_df.tsv'.format(space=info['space'],
                                                                                                               conf=info['conftype'],
                                                                                                               method=method))
                df_concat.to_csv(outfile, sep='\t', index=False)
                
                outfiles.append(outfile)
                
            return outfiles
        
        join_conn_mats = JoinNode(Function(function=write_conn_mat, input_names=['dfs','dfs2','info','group_dir'], 
                                           output_names=['outfiles']), joinsource=infosource, 
                                  joinfield=['dfs','dfs2'], name='join_conn_mats')        
        join_conn_mats.inputs.group_dir = FC_ROI_group_dir
        
        def write_ts(dfs, info, group_dir):
           from itertools import chain
           from misc_funs import flatten
           import pandas as pd
           import os, re
           
           all_files = list(flatten(dfs))
              
           dfs_read = (pd.read_csv(f, sep='\t') for f in all_files)
           df_concat = pd.concat(dfs_read).reset_index()
            
           outfile = os.path.join(group_dir,'space-{space}_conftype-{conf}_desc-timeseries_df.tsv'.format(space=info['space'],
                                                                                                          conf=info['conftype'],))
           df_concat.to_csv(outfile, sep='\t', index=False)
            
           return outfile
        
        join_ts = JoinNode(Function(function=write_ts, input_names=['dfs','info','group_dir'], 
                                    output_names=['outfiles']), joinsource=infosource, 
                           joinfield=['dfs'], name='join_ts')           
        join_ts.inputs.group_dir = FC_ROI_group_dir
        
        # =============================================================================
        # DATASINK
        # =============================================================================
        datasink = Node(DataSink(base_directory=FC_ROI_dir), name="datasink")
        substitutions = [('_ptc_','')]
        datasink.inputs.substitutions=substitutions
        
        # =============================================================================
        # CONNECT
        # =============================================================================
        
        wf_FC_ROI.connect([\
                                (infosource, select_files, [('ptc', 'ptc')]),
                                (select_files, func_info, [('func_json','func_json'),
                                                           ('denoised_func','func_file')]),
                                
                                ###############################
                                # COMPUTE CONNECTIVITY MEASURES  
                                ###############################
                                
                                # RSFC - AFNI
                                (select_files, RSFC_derivatives, [('bold_mask', 'mask_file'),
                                                                  ('denoised_func','in_file')]),
                                (func_info, RSFC_derivatives, [('d', 'func_info')]),
                                (('func_no_bandpass',prefix_add), 'prefix')]),
                                               
                                # reho-kendal - AFNI
                                (select_files, reho_afni_whole_brain, [('bold_mask','mask_file'),
                                                                       ('denoised_func','in_file')]),
                                                             
                                (select_files, get_ts, [('atlas', 'atlas'),
                                                        ('atlas_labels','atlas_labels'),
                                                        ('denoised_func','func_file')]),
                                
                                (infosource, get_ts, [('ptc','ptc')]),
                
                                (get_ts, xcorr, [('df_ts','df_ts')]),
                                (func_info, xcorr, [('d', 'func_info')]),
                                
                                (get_ts, ED, [('df_ts','df_ts')]),
                                (func_info, ED, [('d', 'func_info')]),
                                
                                (get_ts, MD, [('df_ts','df_ts')]),
                                (func_info, MD, [('d', 'func_info')]),
                                
                                (get_ts, WD, [('df_ts','df_ts')]),
                                (func_info, WD, [('d', 'func_info')]),
                                
                                (get_ts, DTW, [('df_ts','df_ts')]),
                                (func_info, DTW, [('d', 'func_info')]),
                                
                                (get_ts, MI, [('df_ts','df_ts')]),
                                (func_info, MI, [('d', 'func_info')]),
                                
                                (get_ts, corr_emp, [('df_ts', 'df_ts')]),   
                                (func_info, corr_emp, [('d', 'func_info')]),
                                
                                (get_ts, corr_LW, [('df_ts', 'df_ts')]),   
                                (func_info, corr_LW, [('d', 'func_info')]),
        
                                (get_ts, parcorr_LW, [('df_ts', 'df_ts')]),  
                                (func_info, parcorr_LW, [('d', 'func_info')]), 
        
                                (get_ts, mscoherence, [('df_ts','df_ts')]),
                                (func_info, mscoherence, [('d', 'func_info')]),                        
                
                                ## For GT metrics, remember to set the which_corr variable!!
                                (corr_LW, centrality_metrics, [('connmat', 'connmat')]),
                                (get_ts, centrality_metrics, [('labels', 'name_rois')]),
                                (func_info, centrality_metrics, [('d', 'func_info')]),
                                 
                                (corr_LW, degree_metrics, [('connmat', 'connmat')]),
                                (get_ts, degree_metrics, [('labels', 'name_rois')]),
                                (func_info, degree_metrics, [('d', 'func_info')]),     
                                
                                # create dataframe with connectivity values (RSFC) - AFNI
                                (select_files, AFNI_conns, [('atlas', 'atlas'),
                                                            ('atlas_labels', 'atlas_labels')]),
                                (RSFC_derivatives, AFNI_conns, [( ('metrics',get_briks), 'RSFC')]),
                                (reho_afni_whole_brain, AFNI_conns, [('out_file', 'ReHoKCC')]),
                                (func_info, AFNI_conns, [('d', 'func_info')]),
                                
                                (infosource, datasink, [('ptc','strip_dir'),
                                                        ('ptc','container')]),
                                
                                (corr_LW, datasink, [('df_corr','{s}.{d}.@f1'.format(s=space,d=name_wf))]),
                                (corr_emp, datasink, [('df_corr','{s}.{d}.@f2'.format(s=space,d=name_wf))]),
                                (parcorr_LW, datasink, [('df_corr','{s}.{d}.@f3'.format(s=space,d=name_wf))]),
                                (xcorr, datasink, [('df_dist','{s}.{d}.@f4'.format(s=space,d=name_wf))]),
                                (ED, datasink, [('df_dist','{s}.{d}.@f5'.format(s=space,d=name_wf))]),
                                (MD, datasink, [('df_dist','{s}.{d}.@f6'.format(s=space,d=name_wf))]),
                                (WD, datasink, [('df_dist','{s}.{d}.@f7'.format(s=space,d=name_wf))]),
                                (DTW, datasink, [('df_dist','{s}.{d}.@f8'.format(s=space,d=name_wf))]),
                                (mscoherence, datasink, [('df_dist','{s}.{d}.@f9'.format(s=space,d=name_wf))]),
                                (MI, datasink, [('df_dist','{s}.{d}.@f11'.format(s=space,d=name_wf))]),                        
                                (centrality_metrics, datasink, [('df_mat','{s}.{d}.@f12'.format(s=space,d=name_wf))]),
                                (degree_metrics, datasink, [('df_mat','{s}.{d}.@f13'.format(s=space,d=name_wf))]),
                                (AFNI_conns, datasink, [('out_values','{s}.{d}.@f14'.format(s=space,d=name_wf))]),
                                (get_ts, datasink, [('df_ts','{s}.{d}.@f15'.format(s=space,d=name_wf))]),
                                
                                (func_info, join_conn_mats, [('d','info')]),
                                (datasink, join_conn_mats, [('out_file','dfs')]),
                                (AFNI_conns, join_conn_mats, [('dfs','dfs2')]),
                                (func_info, join_ts, [('d','info')]),
                                (get_ts, join_ts, [('df_ts','dfs')]),
                                
                                ])
            
        wf_FC_ROI.run()
