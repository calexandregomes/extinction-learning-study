
import numpy as np, pandas as pd, os

from nipype.interfaces import fsl,afni
from nipype import Workflow,Node,Function,MapNode,IdentityInterface,config
from nipype.interfaces.io import SelectFiles,DataSink
from nipype.algorithms.confounds import ACompCor
from nipype.interfaces.base import SimpleInterface,TraitedSpec,CommandLineInputSpec,CommandLine,File,OutputMultiPath,traits,Directory, BaseInterface, BaseInterfaceInputSpec
from subprocess import Popen,PIPE,call


fs_file = pd.read_table(os.path.join(os.getenv('FREESURFER_HOME'),'FreeSurferColorLUT.txt'), sep=r'\s+',
                        comment='#', header=None, names=['fs_Index','Region','R','G','B','A'], engine='python')
tissues = {'WM':  ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter'], 
           'CSF': ['Left-Lateral-Ventricle','Left-Inf-Lat-Vent','3rd-Ventricle','4th-Ventricle','CSF',
                   'Right-Lateral-Ventricle','Right-Inf-Lat-Vent','5th-Ventricle'] }

AG_EXP = {'A02':['Extinction_EEG_fMRI'], 'A03':['3T','3T_EEG'], 'A05':['study3'],
          'A08':['ATOACQ','ContextII','Cort','TIA'], 
          'A09':['Extinction_Generalization_I','Extinction_Generalization_II'],
          'A11':['study1'], 'A12':['study_1','study_2']}

trim_values = [2,None]
desclab = 'trimmed{}'.format(''.join(map(str,trim_values)))

for AG, studies in AG_EXP.items():
    for exp in studies:

        bids_dir = os.path.join('/media/f02/F02', AG, exp)        
        wf_dir = os.path.join(bids_dir, 'workflows')
        rawdata_dir = os.path.join(bids_dir, 'rawdata')
        source_dir = os.path.join(bids_dir, 'sourcedata', 'niftis')
        work_dir = os.path.join(bids_dir, 'work_dir', 'func_denoised')
        derivatives_dir = os.path.join(bids_dir, 'derivatives')
        fmriprep_dir = os.path.join(derivatives_dir, 'fmriprep')
        freesurfer_dir = os.path.join(fmriprep_dir, 'sourcedata','freesurfer')
        func_denoised_dir = os.path.join(derivatives_dir,'func_denoised')
        fs_license = os.path.join(os.getenv('FREESURFER_HOME'),'license.txt')
        crashdir = os.path.join(bids_dir, 'crashfiles')
        
        
        ptc_tsv = pd.read_csv(os.path.join(rawdata_dir, 'participants.tsv'), 
                              sep='\t', usecols=['participant_id','rsfMRI','T1w'])
        participants = ptc_tsv.loc[(ptc_tsv.T1w==1)&(ptc_tsv.rsfMRI==1),'participant_id'].tolist()

        wf_denoising_rest = Workflow(name="wf_denoising_rest", base_dir=wf_dir)
        wf_denoising_rest.config['execution'] = {'use_relative_paths': False,
                                                 'stop_on_first_rerun': True,
                                                 'hash_method': 'content',
                                                 'remove_unnecessary_outputs': False,
                                                 'crashfile_format': 'txt',
                                                 'crashdump_dir': crashdir}
        
        # =============================================================================
        # INFOSOURCE - A FUNCTION FREE NODE TO ITERATE OVER THE LIST OF SUBJECT NAMES
        # =============================================================================
        infosource = Node(IdentityInterface(fields=['ptc']), name="infosource")
        infosource.iterables = [('ptc', participants)]
        
        
        templates = {
            # bold
            'bold_space_T1w': os.path.join(fmriprep_dir,'{ptc}', 'func', '*space-T1w_desc-preproc_bold.nii.gz'),
            'bold_space_MNI152NLin6Asym': os.path.join(fmriprep_dir,'{ptc}', 'func', '*space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'),
            # masks
            'mask_bold_space_T1w': os.path.join(fmriprep_dir,'{ptc}', 'func', '*space-T1w_desc-brain_mask.nii.gz'),
            'mask_bold_MNI152NLin6Asym': os.path.join(fmriprep_dir,'{ptc}', 'func', '*space-MNI152NLin6Asym_desc-brain_mask.nii.gz'),
            # confounds
            # 'AROMA_noise_ICs': os.path.join(fmriprep_dir,'{ptc}', 'func', '*_AROMAnoiseICs.csv'),
            # 'melodic_mixing': os.path.join(fmriprep_dir,'{ptc}', 'func', '*_desc-MELODIC_mixing.tsv'),
            'confounds_file': os.path.join(fmriprep_dir,'{ptc}', 'func', '*_desc-confounds_regressors.tsv'),
            # freesurfer
            'aparc_aseg': os.path.join(fmriprep_dir,'{ptc}', 'func', '*_space-T1w_desc-aparcaseg_dseg.nii.gz'),
            }
        
        select_files = Node(SelectFiles(templates), name='select_files')
        
        def list_noise_ICs(AROMA_noise_ICs):
            import numpy as np
            return list(np.loadtxt(AROMA_noise_ICs,delimiter=',', dtype=int))
        
                       
        fslregfilt = Node(fsl.FilterRegressor(), name="fslregfilt")
        
        def out_file_fslregfilt(ptc):
            out_file = '{ptc}_task-rest_space-T1w_desc-AROMAnonaggr_bold.nii.gz'.format(ptc=ptc)
            return out_file
        
        
        def confounds_extract(ptc,confounds_file):
            import pandas as pd
            import os,json,re
                
            [sub,task,desc,suff] = os.path.basename(confounds_file).split('_')
        
            def create_conf(df, name):
                header = {name:df.columns.to_list()}
                filename = '_'.join([sub,task,name,desc,suff])
                comp_file = os.path.abspath(filename)
                df.fillna(0).to_csv(comp_file, sep='\t', index=None)
                return comp_file, header
                
            f = pd.read_csv(confounds_file, sep='\t') 
            
            # only cosine
            df_cosine = f.filter(regex='^(cosine.*|non_steady_state_outlier.*)$')
            f_cosine, cosine_hdr = create_conf(df_cosine,'conftype-cosine')
            
            # CSF + WM
            df_CSF_WM = f.filter(regex='^(csf|white_matter|non_steady_state_outlier.*)$')
            f_CSF_WM, CSF_WM_hdr = create_conf(df_CSF_WM,'conftype-WMCSF')
            
            df_CSF_WM_exp = f.filter(regex='^(csf.*|white_matter.*|non_steady_state_outlier.*)$')
            f_CSF_WM_exp, CSF_WM_exp_hdr = create_conf(df_CSF_WM_exp,'conftype-WMCSFExp')
            
            # 6 MP + CSF + WM
            df_basic = f.filter(regex='(rot|trans)_.$|^(csf|white_matter|non_steady_state_outlier.*)$')  
            f_basic, basic_hdr = create_conf(df_basic,'conftype-Basic')
            
            # 6 MP + CSF + WM + global signal
            df_basic_global = f.filter(regex='(rot|trans)_.$|^(csf|white_matter|global_signal|non_steady_state_outlier.*)$')  
            f_basic_global, basic_global_hdr = create_conf(df_basic_global,'conftype-BasicGlobal')
        
            # 24 MP + 4 CSF + 4 WM ()
            df_basic_exp = f.filter(regex='^(rot|trans|csf|white_matter|non_steady_state_outlier.*).*$')  
            f_basic_exp, basic_exp_hdr = create_conf(df_basic_exp,'conftype-BasicExp')
            
            # 24 MP + 4 CSF + 4 WM + 4 global signal (36 regressors)  
            df_satterthwaite = f.filter(regex='^(csf|white_matter|global|rot|trans|non_steady_state_outlier.*).*$')
            f_satterthwaite, satterthwaite_hdr = create_conf(df_satterthwaite,'conftype-Satterthwaite')
            
            # 24 MP + N CompCor + 6 cosine 
            df_compcor = f.filter(regex='^(rot|trans|a_comp_cor|cosine|non_steady_state_outlier.*).*$')
            f_compcor, compcor_hdr = create_conf(df_compcor,'conftype-CompCor')
                
            ds = {**cosine_hdr,**CSF_WM_hdr,**CSF_WM_exp_hdr,**basic_hdr,**basic_global_hdr,
                  **basic_exp_hdr,**satterthwaite_hdr,**compcor_hdr}
            json_confounds = os.path.abspath('_'.join([sub,task,'desc-confoundsCols','regressors.json']))
            with open(json_confounds, 'w') as filename:
                json.dump(ds, filename, indent=4)
            
            confounds_list = [f_cosine, f_CSF_WM, f_CSF_WM_exp, f_basic, f_basic_global, 
                              f_basic_exp, f_satterthwaite, f_compcor]
            
            get_conftype =  lambda x: re.match(r'.*_conftype\-([^\_]+)', x).group(1) 
        
            nii_filenames = [os.path.basename('conftype-'+get_conftype(x))+'.nii.gz' 
                             for x in confounds_list]
            
            return confounds_list, nii_filenames, json_confounds
            
        extract_confounds = Node(Function(function=confounds_extract, input_names=['ptc','confounds_file'], 
                                          output_names=['confounds_list','nii_filenames','json_confounds']),
                                 name='extract_confounds')
        
        # =============================================================================
        # PERFORM aCOMPCOR
        # =============================================================================
        def get_tissues(aparc_aseg, fs_file, tissues):
            import nibabel as nib, numpy as np, os
            return_files = []
            img = nib.load(aparc_aseg)
            data = img.get_fdata()
            ind_array = {k:fs_file.loc[fs_file['Region'].isin(v)].fs_Index.values for k,v in tissues.items()}
            for k,v in ind_array.items():
                tissue = np.isin(data,ind_array.get(k).astype(int))
                img_new = nib.nifti1.Nifti1Image(tissue, None, header=img.header.copy())
                filename = os.path.abspath('%s_mask.nii'%k)
                return_files.extend([filename])
                nib.save(img_new, filename) 
            return return_files
        
        extract_tissues = Node(Function(function=get_tissues, input_names=['aparc_aseg','fs_file','tissues'], 
                                               output_names=['return_files']), name='extract_tissues')
        extract_tissues.inputs.tissues = tissues
        extract_tissues.inputs.fs_file = fs_file
        
        erode_imgs = MapNode(fsl.maths.ErodeImage(), iterfield=['in_file'], name='erode_img')
        
        acompcor = Node(ACompCor(pre_filter='cosine', save_pre_filter=True, save_metadata=True, 
                                 failure_mode='NaN', variance_threshold=0.50, ignore_initial_volumes=2), 
                        name='acompcor')
        acompcor.iterables = [("merge_method", ['union','none'])]
        
        def rm_header(file):
            import pandas as pd, os
            from nipype.utils.filemanip import split_filename
            [base,_,_] = split_filename(file)
            comp_file = os.path.join(base,'components_file_no_header.txt')
            f = pd.read_csv(file, sep='\n')
            f.to_csv(comp_file, header=None, index=None)
            return comp_file
        
        
        def gen_confounds_file(comp_file, fmriprep_confs):    
            d = locals()
            
            import numpy as np, pandas as pd
            import os
        
            list_reg = [pd.read_csv(v,sep='\t', skiprows=0) for k,v in d.items()]
            
            concat_file = os.path.abspath('nuisance_regressors.txt')
            concat_files = pd.concat(list_reg,axis=1)
            if not (concat_files==1).all().any():
                concat_files = np.insert(concat_files,concat_files.shape[1]+1,[1],axis=1)
            concat_files.to_csv(concat_file, index=None, sep='\t')
            return concat_file
            
        confounds_file = MapNode(Function(function=gen_confounds_file, 
                                          input_names=['comp_file','fmriprep_confs'],
                                          output_names=['concat_file']), 
                                 iterfield=['fmriprep_confs'], name='confounds_file')
              
        def trim_func_data(func_file, t):
            import os
            import nibabel as nib
            from misc_funs import split_filename,replaceValue
        
            func_im = nib.load(func_file)    
            basedir,filename,ext = split_filename(func_file)
        
            if t:
                new_func = func_im.slicer[..., t[0]:t[1]]
                desclab = 'trimmed'+''.join(map(str, t))
                new_filename = replaceValue(func_file, ['desc',desclab])
            else:
                new_func = func_im
                new_filename = filename+ext
                
            outfile = os.path.join(basedir, new_filename+ext)
            nib.save(new_func, outfile)
            return outfile
        
        trim_func = MapNode(Function(function=trim_func_data, input_names=['func_file','t'],
                                     output_names=['outfile']), iterfield=['func_file'],
                            name='trim_func_{}'.format(''.join(map(str,trim_values))))
        trim_func.inputs.t = trim_values
        
        # =============================================================================
        # PERFORM NUISANCE REGRESSION
        # =============================================================================
        polort = 2
        low_cutoff = 0.005
        high_cutoff = 0.1
        bandpass = [(0.0,1.0), (low_cutoff,1.0), (low_cutoff,high_cutoff)]
        demean = True
        
        nuisreg_AROMA_T1w_afni = Node(afni.TProject(polort=polort, automask=False,
                                                    out_file='confounds-ICAAROMA_desc-afni_bold.nii.gz'), 
                                             name='nuisreg_AROMA_T1w_afni')
        # nuisreg_AROMA_T1w_afni.iterables = [('bandpass', bandpass)]
        
        nuisreg_AROMA_T1w_fsl = Node(fsl.GLM(demean=demean, 
                                             out_res_name='confounds-ICAAROMA_desc-fsl_bold.nii.gz'),
                                     name='nuisreg_AROMA_T1w_fsl')
        
        
        # Perform nuisance regression with 3dTProject
        nuisreg_T1w_afni = MapNode(afni.TProject(polort=polort, automask=False), 
                                   iterfield=['ort','out_file'], name='nuisreg_T1w_afni')
        # nuisreg_T1w_afni.iterables = [('bandpass', bandpass)]
        nuisreg_MNI152NLin6Asym_afni = nuisreg_T1w_afni.clone(name='nuisreg_MNI152NLin6Asym_afni')
        
        
        # Perform nuisance regression with FSL
        nuisreg_T1w_fsl = MapNode(fsl.GLM(demean=demean, out_res_name='denoised_func.nii.gz'),
                                  iterfield=['design','out_res_name'], name='nuisreg_T1w_fsl')
        nuisreg_MNI152NLin6Asym_fsl = nuisreg_T1w_fsl.clone(name='nuisreg_MNI152NLin6Asym_fsl')
            
        # =============================================================================
        # DATASINK
        # =============================================================================
        datasink_conf_T1w = Node(DataSink(parameterization=True, base_directory=func_denoised_dir),
                                 name="datasink_conf_T1w")
        substitutions = [('_bandpass_','bandpass-'),('_ptc_','')]
        datasink_conf_T1w.inputs.substitutions=substitutions
        datasink_conf_T1w.inputs.regexp_substitutions = [('_nuisreg_.*/',''), 
                                                          ('_merge_method_*','mergemethod-')]
        
        datasink_conf_MNI152NLin6Asym = datasink_conf_T1w.clone(name="datasink_conf_MNI152NLin6Asym")
        
        datasink_AROMA_T1w = datasink_conf_T1w.clone(name="datasink_AROMA_T1w")
        
        
        wf_denoising_rest.connect([\
        
                        (infosource, select_files, [('ptc', 'ptc')]),
                        
                        (select_files, fslregfilt, [('melodic_mixing','design_file'),
                                                    ( ('AROMA_noise_ICs',list_noise_ICs), 'filter_columns'),
                                                    ('bold_space_T1w','in_file')]),                
                        (infosource, fslregfilt, [( ('ptc',out_file_fslregfilt), 'out_file')]),
                        
                        (select_files, extract_confounds, [('confounds_file','confounds_file')]),
                        (infosource, extract_confounds, [('ptc','ptc')]),
        
                        
                        #aCompCor
                        (select_files, extract_tissues, [('aparc_aseg', 'aparc_aseg')]),
                        (extract_tissues, erode_imgs, [('return_files', 'in_file')]),
                        (erode_imgs, acompcor, [('out_file', 'mask_files')]),
                        (fslregfilt, acompcor, [('out_file', 'realigned_file')]),
                        
                        ############################################################
                        #################### NUISANCE REGRESSION ###################
                        ############################################################
        
                        ##############################################################
                        # WITH ICA-AROMA (RECALCULATED WM AND CSF) - T1w
                        (fslregfilt, nuisreg_AROMA_T1w_afni, [('out_file','in_file')]),
                        (acompcor, nuisreg_AROMA_T1w_afni, [( ('components_file',rm_header), 'ort')]),
                        (select_files, nuisreg_AROMA_T1w_afni, [('mask_bold_space_T1w','mask')]),
        
                        (acompcor, confounds_file, [('components_file','comp_file')]),
                        (extract_confounds, confounds_file, [('confounds_list','fmriprep_confs')]),
                        (fslregfilt, nuisreg_AROMA_T1w_fsl, [('out_file','in_file')]),
                        (confounds_file, nuisreg_AROMA_T1w_fsl, [('concat_file', 'design')]),
                        (select_files, nuisreg_AROMA_T1w_fsl, [('mask_bold_space_T1w','mask')]),
                        ##############################################################
         
                        ##############################################################            
                        # WITHOUT ICA-AROMA - T1w
                        (extract_confounds, nuisreg_T1w_afni, [('confounds_list','ort'),
                                                               ('nii_filenames','out_file')]),
                        (select_files, nuisreg_T1w_afni, [('bold_space_T1w','in_file'),
                                                          ('mask_bold_space_T1w','mask')]),
        
                        (extract_confounds, nuisreg_T1w_fsl, [('confounds_list','design'),
                                                              ('nii_filenames','out_res_name')]),
                        (select_files, nuisreg_T1w_fsl, [('bold_space_T1w','in_file'),
                                                         ('mask_bold_space_T1w','mask')]),                   
                        
                        (nuisreg_T1w_fsl, trim_func, [('out_res','func_file')]),
                        
                        ############################################################
                        ######################### DATASINK #########################
                        ############################################################              
                        (infosource, datasink_AROMA_T1w, [('ptc','strip_dir'),
                                                          ('ptc','container')]),
                        (nuisreg_AROMA_T1w_fsl, datasink_AROMA_T1w, [('out_file','ICA_AROMA')]),
                        
                        (infosource, datasink_conf_T1w, [('ptc','strip_dir'),
                                                          ('ptc','container')]),
                        (trim_func, datasink_conf_T1w, [('outfile','denoised_func.space-T1w')]),
                        (extract_confounds, datasink_conf_T1w, [('confounds_list','confounds.@c1')]),
                        (extract_confounds, datasink_conf_T1w, [('json_confounds','confounds.@c2')]),                                        
                        ])
        res_denoise_rest = wf_denoising_rest.run('MultiProc')
        
        def create_BIDS_filenames(ptc,denoised_dir=func_denoised_dir):
            import glob,os,shutil
            from misc_funs import split_filename
        
            subfolders = glob.glob(os.path.join(denoised_dir,ptc,'*/'))
            for sf in subfolders:
                os.chdir(sf)
                subdirs = [d for d in os.listdir(sf) if os.path.isdir(os.path.join(sf,d))]
                if not subdirs:
                    continue
                
                files = [glob.glob(os.path.join(sd,'**/*.nii.gz'),recursive=True) for sd in subdirs][0]
                for f in files:
        
                    split_f = split_filename(f)
                    all_labs = '_'.join(split_f[:-1]) #exclude extension
                    bids_filename = '_'.join([ptc, 'task-rest', all_labs]) + '_bold' + split_f[-1]
        
                    # print(os.path.join(sf,bids_filename))
                    new_f = os.path.join(sf,bids_filename)
                    try: 
                        os.remove(new_f)
                    except OSError:
                        pass
                    shutil.move(os.path.join(sf,f), new_f)
                       
                for subdir in glob.glob(os.path.join(sf,'*/')):
                    files_subdir = glob.glob(os.path.join(subdir,'**/*'),recursive=True)
                    leftover_files = [f for f in files_subdir if os.path.isfile(f)]
                    if not leftover_files:
                        shutil.rmtree(subdir)
                    else:
                        print('These files have not been moved: {}'.format(leftover_files))
        
        print('\n\nConverting filenames to BIDS...\n')
        [create_BIDS_filenames(p) for p in participants]
