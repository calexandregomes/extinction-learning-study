
# NEED TO HAVE RUN:
#   F02_pipeline_BIDS
#   F02_pipeline_fmriprep
#   F02_pipeline_ROIs
#   F02_pipeline_SUIT

import numpy as np, pandas as pd, os, glob, ast

from nipype.interfaces import fsl,ants,utility
from nipype import Workflow,Node,Function,MapNode,IdentityInterface,JoinNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.freesurfer import MRIsConvert,MRIConvert,BBRegister,Label2Vol,Label2Label,Binarize,Tkregister2
import nipype.interfaces.mrtrix3 as mrtrix
from nipype.interfaces.base import TraitedSpec,CommandLineInputSpec,File,OutputMultiPath,traits,Directory, SEMLikeCommandLine
from nipype.interfaces.workbench.base import WBCommand
from nipype.interfaces.base import TraitedSpec,CommandLineInputSpec,CommandLine, \
    File,OutputMultiPath,traits, Directory, BaseInterface, BaseInterfaceInputSpec, isdefined, InputMultiPath
from nipype.interfaces.freesurfer.base import FSTraitedSpec,FSCommand
from nipype.interfaces.fsl.base import FSLCommandInputSpec, FSLCommand
from subprocess import call
from misc_funs import exclude_subs, split_equal

fs_file = pd.read_csv(os.path.join(os.getenv('FREESURFER_HOME'),'FreeSurferColorLUT.txt'), sep=r'\s+',
                        comment='#', header=None, names=['fs_Index','Region','R','G','B','A'], engine='python')

AG_EXP = {'A03':['3T','3T_EEG'], 'A05':['study3'],
          'A08':['ATOACQ','ContextII','Cort','TIA'], 
          'A09':['Extinction_Generalization_I','Extinction_Generalization_II'],
          'A12':['study_1','study_2']}

name_wf_DTI = 'wf_DTI'


for AG, studies in AG_EXP.items():
    for exp in studies:
        # Specify directories and participant list
        base_dir = '/media/f02/F02'
        bids_dir = os.path.join(base_dir, AG, exp)
        wf_dir = os.path.join(bids_dir, 'workflows')
        rawdata_dir = os.path.join(bids_dir, 'rawdata')
        source_dir = os.path.join(bids_dir, 'sourcedata')
        derivatives_dir = os.path.join(bids_dir, 'derivatives')
        fmriprep_dir = os.path.join(derivatives_dir, 'fmriprep')
        freesurfer_dir = os.path.join(derivatives_dir, 'freesurfer')
        ROIs_dir = os.path.join(derivatives_dir, 'ROIs')
        DTI_dir = os.path.join(derivatives_dir, 'DTI')
        group_DTI_dir = os.path.join(DTI_dir, 'group')
        if not os.path.exists(group_DTI_dir): 
            os.makedirs(group_DTI_dir)
        crashdir = os.path.join(bids_dir, 'crashfiles')
                                                                                                               
        n_threads = 20
        
        participants_file = pd.read_csv(os.path.join(rawdata_dir, 'participants.tsv'),
                                        sep='\t', usecols=['participant_id','dwi', 'T1w'])
        participants = participants_file.loc[participants_file.loc[:,['dwi','T1w']].all(axis=1),
                                             'participant_id'].tolist()
        exc_subs_file = os.path.join(base_dir, 'exclude_subs.csv')
        participants = exclude_subs(exc_subs_file, participants, AG, exp, pipeline='DTI')
        
        wf_DTI = Workflow(name=name_wf_DTI, base_dir=wf_dir)
        wf_DTI.config['execution'] =  {'stop_on_first_crash': False,
                                       'hash_method': 'content',
                                       'use_relative_paths': False,
                                       #'keep_inputs': True, 
                                       'remove_unnecessary_outputs': False,
                                       'crashfile_format': 'txt',
                                       'crashdump_dir': crashdir}
        
        ###############################################################################
        ######################## SELECT SUBJECTS/FILES ################################
        ###############################################################################
        
        # =============================================================================
        # INFOSOURCE - A FUNCTION FREE NODE TO ITERATE OVER THE LIST OF SUBJECT NAMES
        # =============================================================================
        infosource = Node(IdentityInterface(fields=['ptc']), name="infosource")
        infosource.iterables = [('ptc', participants)]
        
        templates = {'dwi': os.path.join(rawdata_dir,'{ptc}', 'dwi', '*dwi*.nii.gz'),
                     'dwi_json': os.path.join(rawdata_dir,'{ptc}', 'dwi', '*dwi*.json'),
                     'oppPE': os.path.join(rawdata_dir,'{ptc}', 'fmap', '*epi*.nii.gz'),
                     'oppPE_json': os.path.join(rawdata_dir,'{ptc}', 'fmap', '*epi*.json'),
                     'bvals': os.path.join(rawdata_dir,'{ptc}', 'dwi', '*.bval'),
                     'bvecs': os.path.join(rawdata_dir,'{ptc}', 'dwi', '*.bvec'),
                     'anat': os.path.join(rawdata_dir,'{ptc}', 'anat', '*T1w*.nii.gz'),
                     'T1wmask': os.path.join(fmriprep_dir,'{ptc}', 'anat', '*desc-brain_mask.nii.gz'),
                     'orig': os.path.join(freesurfer_dir,'{ptc}', 'mri', 'orig.nii.gz'),
                     'FSmask': os.path.join(freesurfer_dir,'{ptc}', 'mri', 'brainmask.mgz')}
        
        select_files = Node(SelectFiles(templates), name='select_files')
        
        def fmriprep_files(derivatives_dir, ptc):
            import numpy as np, os, glob
            p_fmriprep = os.path.join(derivatives_dir, 'fmriprep', ptc)
            p_freesurfer = os.path.join(derivatives_dir, 'freesurfer', ptc)
            
            T1w = glob.glob(os.path.join(p_fmriprep,'**/{}_desc-preproc_T1w.nii.gz').format(ptc),recursive=True)[0]
            
            fs2T1w_reg = glob.glob(os.path.join(p_fmriprep,'**/*from-fsnative_to-T1w*'),recursive=True)[0]
            
            mask_MNI152NLin6Asym = glob.glob(os.path.join(p_fmriprep,'**/*space-MNI152NLin6Asym_desc-brain_mask.nii.gz'),recursive=True)[0]
        
            h5_from_MNI152NLin6Asym_to_T1w = glob.glob(os.path.join(p_fmriprep,'**/*from-MNI152NLin6Asym_to-T1w*.h5'),recursive=True)[0]
            h5_from_T1w_to_MNI152NLin6Asym = glob.glob(os.path.join(p_fmriprep,'**/*from-T1w_to-MNI152NLin6Asym*.h5'),recursive=True)[0]
        
            #freesurfer files
            aparc_aseg_T1w = glob.glob(os.path.join(p_fmriprep,'**/anat/*_desc-aparcaseg_dseg.nii.gz'),recursive=True)[0]
            orig = glob.glob(os.path.join(p_freesurfer,'**/*orig.mgz'),recursive=True)[0]
            annot_aparc2009 = glob.glob(os.path.join(p_freesurfer,'**/*aparc.annot'),recursive=True)
        
            
            return (T1w, fs2T1w_reg, aparc_aseg_T1w, orig, mask_MNI152NLin6Asym, 
                    h5_from_T1w_to_MNI152NLin6Asym, h5_from_MNI152NLin6Asym_to_T1w,
                    annot_aparc2009, p_freesurfer)
        
        get_fmriprep_files = Node(Function(function=fmriprep_files, input_names=['derivatives_dir','ptc'],
                                       output_names=['T1w', 'fs2T1w_reg', 'aparc_aseg_T1w','orig', 'mask_MNI152NLin6Asym',
                                                     'h5_from_T1w_to_MNI152NLin6Asym','h5_from_MNI152NLin6Asym_to_T1w',
                                                     'annot_aparc2009', 'p_freesurfer']), 
                                  name='get_fmriprep_files')
        get_fmriprep_files.inputs.derivatives_dir = derivatives_dir               
        
        ###############################################################################
        ########################### PREPROCESSING #####################################fsleyes
        ###############################################################################
        
        # =============================================================================
        # TELL FSL TO GENERATE OUTPUT AS .NII FILES
        # =============================================================================
        
        joinfiles = Node(utility.Merge(2), name="joinfiles")
        joinfiles.overwrite = False
        fsl.FSLCommand.set_default_output_type('NIFTI')
        denoise =  Node(mrtrix.DWIDenoise(nthreads=n_threads), name='denoise')
        
        gibbs =  Node(mrtrix.MRDeGibbs(nthreads=n_threads), name='gibbs_AP')

        get_B0 = MapNode(fsl.ExtractROI(t_min=0, t_size=1), iterfield=['in_file'], name="get_B0")
        
        def data_after_gibbs(image,dwi_raw_file):
            import os
            import nibabel as nib
            from nilearn.image import index_img
            
            dwi_raw = nib.load(dwi_raw_file)
            B0_PA_vol_nb = dwi_raw.shape[3]
            B0s = index_img(image,[0,B0_PA_vol_nb])
            B0_file = os.path.abspath('B0s_AP_PA.nii.gz')
            B0s.to_filename(B0_file)
        
            dwi = index_img(image,slice(0,B0_PA_vol_nb))
            dwi_file = os.path.abspath('dwi_denoised_gibbed.nii.gz')
            dwi.to_filename(dwi_file)
            
            return B0_file, dwi_file
        
        get_data_after_gibbs = Node(Function(function=data_after_gibbs, input_names=['image','dwi_raw_file'], 
                                     output_names=['B0_file','dwi_file']), name='get_data_after_gibbs')
        
        merge_AP_PA = Node(fsl.Merge(dimension='t'), name="merge_AP_PA")
        
        def get_index_acqparams_files(dwi_file, dwi_json, opp_json, APPA):
            import json, os
            import nibabel as nib, numpy as np
            
            dwi_raw = nib.load(dwi_file)
            dwi_vol_nb = dwi_raw.shape[3]
            
            dAPPA = nib.load(APPA)
            dappa_vol_nb = dAPPA.shape[3]
            
            with open(dwi_json, 'r+') as json_file:
                data = json.load(json_file)
                trot_dwi = data['TotalReadoutTime']
                PE_dir_dwi = data['PhaseEncodingDirection']
            
            with open(opp_json, 'r+') as json_file:
                data = json.load(json_file)
                trot_opp = data['TotalReadoutTime']
                PE_dir_opp = data['PhaseEncodingDirection']
              
            acqparams = np.empty([2,4])
            if PE_dir_dwi=='j':
                acqparams[0] = [0, 1, 0, trot_dwi]
            elif PE_dir_dwi=='j-':
                acqparams[0] = [0, -1, 0, trot_dwi]     
            if PE_dir_opp=='j':
                acqparams[1] = [0, 1, 0, trot_opp]
            elif PE_dir_opp=='j-':
                acqparams[1] = [0, -1, 0, trot_opp]
            
            acqparams_file = os.path.abspath('acqparams.txt')
            np.savetxt(acqparams_file, acqparams, fmt='%d %d %d %1.3f')
            
            index = [1]*dwi_vol_nb
            index_file = os.path.abspath('index.txt')
            np.savetxt(index_file, index, fmt='%d')
            
            return acqparams_file, index_file
        
        acqparams_index_files = Node(Function(function=get_index_acqparams_files, input_names=['dwi_file','dwi_json','opp_json','APPA'], 
                                     output_names=['acqparams_file','index_file']), name='acqparams_index_files')
        
            
        topup = Node(fsl.TOPUP(), name="topup")
        topup.inputs.config = os.path.join(os.getenv('FSLDIR'),'etc/flirtsch/b02b0.cnf')
        
        mean_image = Node(fsl.MeanImage(), name='mean_image')
        
        bet = Node(fsl.BET(mask=True, frac=0.2), name='bet')
        
        bias_correct = Node(mrtrix.DWIBiasCorrect(nthreads=n_threads, use_fsl=True,
                                                  out_file='eddy_biascorr.nii.gz'), name='bias_correct')
        
        eddy = Node(fsl.Eddy(repol=True, num_threads=n_threads, cnr_maps=True, 
                             residuals=True, out_base='eddy'), name='eddy')
        
        def base_name(file,base_name='eddy'):
            import os
            return os.path.join(os.path.dirname(file),base_name)
            
        eddy_quad = Node(fsl.EddyQuad(output_dir='data_quad') ,name='eddy_quad')   
        
		dtifit = Node(fsl.DTIFit(),name='dtifit')
        
        # =============================================================================
        # perform segmentation using freesurfer's BBRegister tool
        # =============================================================================
        
        get_B0_eddy = Node(fsl.ExtractROI(t_min=0, t_size=1), name="get_B0_eddy")
        
        bbreg = Node(BBRegister(init='fsl', contrast_type='t1', registered_file='reg.nii.gz',
                                out_fsl_file=True), name='bbreg')
        bbreg.inputs.subjects_dir = os.path.join(derivatives_dir, 'freesurfer')
                
        FS2FA_mat = Node(fsl.ConvertXFM(invert_xfm=True, out_file='FS2FA.mat'), name='FS2FA_mat')
        
        # Break h5 ANTS file 
        disassemble_h5_T1w2MNI = Node(ants.CompositeTransformUtil(process='disassemble'), name='disassemble_h5_T1w2MNI')
        disassemble_h5_MNI2T1w = Node(ants.CompositeTransformUtil(process='disassemble'), name='disassemble_h5_MNI2T1w')
        
        class itk2fslRegInputSpec(CommandLineInputSpec):
            reference_file = File(exists=True, argstr="-ref %s", position=1)
            source_file = File(exists=True, argstr="-src %s", position=2)
            itk_file = File(exists=True, argstr="-itk %s", position=3)
            out_reg = traits.Either(traits.Bool, File(), hash_files=False, desc="Export FSL transform",
                                          argstr="-o %s", position=5)
            ras2fsl = traits.Bool(argstr="-ras2fsl", position=4)
        
        class itk2fslRegOutputSpec(TraitedSpec):
            out_reg = File(exists=True)
        
        class itk2fslReg(SEMLikeCommandLine):
            input_spec = itk2fslRegInputSpec
            output_spec = itk2fslRegOutputSpec
            _cmd = "c3d_affine_tool"
            _outputs_filenames = {"out_reg": "affine.mat"}
        
        itk2fsl_FS2T1w_reg = Node(itk2fslReg(ras2fsl=True, out_reg='fs2anat.mat'), name='itk2fsl_FS2T1w_reg')
        
        itk2fsl_T1w2MNI_reg = Node(itk2fslReg(ras2fsl=True, out_reg='T1w2MNI.mat'), name='itk2fsl_T1w2MNI_reg')
        itk2fsl_T1w2MNI_reg.inputs.reference_file = os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_1mm_brain.nii.gz')
        itk2fsl_T1w2MNI_reg.inputs.reference_file = os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain.nii.gz')
        
        mriconvert = Node(MRIConvert(out_file='orig.nii.gz', out_type='niigz'), name='mriconvert')
        
        FA2anat_mat = Node(fsl.ConvertXFM(concat_xfm=True, out_file='FA2anat.mat'), name='FA2anat_mat')
        anat2FA_mat = Node(fsl.ConvertXFM(invert_xfm=True, out_file='anat2FA.mat'), name='anat2FA_mat')
        
        FA2MNI_mat = Node(fsl.ConvertXFM(concat_xfm=True, out_file='FA2MNI.mat'), name='FA2MNI_mat')
        
        FA2anat_regcheck = Node(fsl.FLIRT(out_file='FA2T1w.nii.gz'), name='FA2anat_regcheck') 
        
        FS2anat_tk = Node(Tkregister2(noedit=True, reg_header=True, fsl_out='FS2anat_fsl.mat', reg_file='FS2anat_fs.dat'), 
                          name='FS2anat_tk')
        
        
        class WBConvertWarpInputSpec(CommandLineInputSpec):
            from_itk = File(exists=True, argstr="-from-itk %s", position=0)
            to_fnirt = traits.Bool(argstr="-to-fnirt",position=1)
            out_file = File(name_source=["from_itk"], name_template='%s_TO_FSL.nii.gz', argstr="%s", position=2)  
            src_file = File(exists=True, argstr="%s", position=3)  
        
        class WBConvertWarpOutputSpec(TraitedSpec):
            out_file = File(exists=True)
        
        class WBConvertWarp(WBCommand):
            input_spec = WBConvertWarpInputSpec
            output_spec = WBConvertWarpOutputSpec
            _cmd = "wb_command -convert-warpfield"
              
        convert_warp = Node(WBConvertWarp(to_fnirt=True), name='convert_warp')
        
        # COMBINE AFFINE WITH WARP 
        FA2MNI_warp = Node(fsl.ConvertWarp(out_file='FA2MNI_warp.nii.gz'), name='FA2MNI_warp')
        FA2MNI_warp.inputs.reference = os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain.nii.gz') 
        
        labels2T1w = MapNode(Label2Label(registration_method='volume'), iterfield=['srclabel'], 
                             name='labels2T1w')
        def arg_reg(reg): return '--reg {}'.format(reg)
        
        labels = ['lh.caudalanteriorcingulate.label','rh.caudalanteriorcingulate.label',
                  'lh.medialorbitofrontal.label','rh.medialorbitofrontal.label']
        
        def get_label(label, fs_dir):
            import os
            labelname = label.split('.label')[0] + 'T1w.label'
            srclabel = os.path.join(fs_dir, 'labels', label)
            trglabel = os.path.join(fs_dir, 'labels', labelname)
            return srclabel, trglabel
            
        labels2T1w_args = MapNode(Function(function=get_label, input_names=['label','fs_dir'], 
                                           output_names=['srclabel', 'trglabel']), 
                                  iterfield=['label'], name='labels2T1w_args')
        
        #bedpostx
        bedpostx = Node(fsl.BEDPOSTX5(n_fibres=3, burn_in=1000, n_jumps=1250, 
                                      sample_every=25, model=2, fudge=1), name='bedpostx')
        
        
        # =============================================================================
        # binarise left and right ROIs from label2vol to be passed to probtrackx2
        # =============================================================================
        
        def get_ROIs(ptc, ROIs_dir, space, rois_vol=[], rois_surf=[]):
            import glob, os
            from misc_funs import flatten
            p = os.path.join(ROIs_dir, ptc, space)
            rv = [glob.glob(os.path.join(p,'*_roi-{}_*.nii.gz'.format(r))) for r in rois_vol]
            rs = [glob.glob(os.path.join(p,'*_roi-{}_*.gii'.format(r))) for r in rois_surf]
            ROIs = [f for f in flatten([rv,rs]) if not '_hemi-Bilateral_' in f]
            
            seed_file = os.path.abspath('seeds.txt')
            with open(seed_file,'w') as outfile:
                outfile.write('\n'.join(ROIs))
                outfile.write('\n')
        
            return ROIs, seed_file
            
        ROIs_seed_fs = Node(Function(function=get_ROIs, input_names=['ptc','ROIs_dir','space','rois_vol','rois_surf'], 
                                     output_names=['ROIs','seed_file']), name='ROIs_seed_fs2')
        ROIs_seed_fs.inputs.rois_surf = ['dACC','vmPFC']
        ROIs_seed_fs.inputs.ROIs_dir = ROIs_dir
        ROIs_seed_fs.inputs.space = 'freesurfer'
        
        ROIs_seed_anat = ROIs_seed_fs.clone('ROIs_seed_anat')
        ROIs_seed_anat.inputs.rois_vol = ['Hippocampus','Amygdala','CerebellumNuclei','dACC','vmPFC',
                                          'LateralVentricle','3rdVentricle','4thVentricle','CSF']
        ROIs_seed_anat.inputs.rois_vol = ['LateralVentricle','4thVentricle',
                                          'iOccipital','lOccipital','Supramarginal','Precentral','Lingual']
        
        ROIs_seed_anat.inputs.space = 'anat'        
        diffusion_ROIs = MapNode(fsl.FLIRT(interp='nearestneighbour', apply_xfm=True), iterfield=['in_file'], 
                                 name='diffusion_ROIs')
        
        
        # prepare files for tractography
        def get_files_tract(ptc, ROIs_dir, space, tract_d):
            import glob, os, ast
            from misc_funs import flatten
            p = os.path.join(ROIs_dir, ptc, space)
                
            seed = list(flatten([glob.glob(os.path.join(p,'{}'.format(r))) for r in tract_d['seed']]))
            target = list(flatten([glob.glob(os.path.join(p,'{}'.format(r))) for r in tract_d['target']]))
            stop = list(flatten([glob.glob(os.path.join(p,'{}'.format(r))) for r in tract_d['stop']]))
            waypoint = list(flatten([glob.glob(os.path.join(p,'{}'.format(r))) for r in tract_d['waypoint']]))
            avoid = list(flatten([glob.glob(os.path.join(p,'{}'.format(r))) for r in tract_d['avoid']]))
            
            d = {}
            for n,f in [('seed',seed),('stop',stop),('waypoint',waypoint),('avoid',avoid)]:
                filename = os.path.abspath('{}.txt'.format(n))
                with open(filename,'w') as outfile:
                    outfile.write('\n'.join(f))
                    outfile.write('\n')
                    d.update({n:filename})
            d.update({'target':target})            
            return d
        
        files_tract = MapNode(Function(function=get_files_tract, input_names=['ptc','ROIs_dir','space','tract_d'], 
                                       output_names=['d']), 
                              iterfield=['tract_d'], name='files_tract2')
        files_tract.inputs.space = 'freesurfer'
        files_tract.inputs.ROIs_dir = ROIs_dir
        
        dict_tract = os.path.join(base_dir, 'test_dict.txt')
        dict_tract = os.path.join(base_dir, 'dict_tractography2.txt')
        with open(dict_tract, 'rb') as td:
            tract_d = [ast.literal_eval(l.decode()) for l in td if (l.strip() and not l.startswith(b'#'))]
        files_tract.inputs.tract_d = tract_d
        
        
        # =============================================================================
        # PROBTRACKX2       
        # =============================================================================
        
        mriconvert_brainmask = Node(MRIConvert(), name='mriconvert_brainmask')
        def mriconvert_outfile(filename):
            import os
            from misc_funs import split_filename
            [base,name,_] = split_filename(filename, exts='.mgz')
            return os.path.join(base, name+'.nii.gz')
        
        
        class probtrackFSInputSpec(BaseInterfaceInputSpec):
            ds = traits.List()
            mask = File()
            thsamples = InputMultiPath(File(exists=True), mandatory=True)
            fsamples = InputMultiPath(File(exists=True), mandatory=True)
            phsamples = InputMultiPath(File(exists=True), mandatory=True)
            seed_ref = File()
            xfm = File()
            options  = traits.Dict({}, usedefault=True)
        
        class probtrackFSOutputSpec(TraitedSpec):
            result = traits.Either(traits.Dict(), traits.List())
        
        class probtrackFS(BaseInterface):
            input_spec = probtrackFSInputSpec
            output_spec = probtrackFSOutputSpec
            
            def _run_interface(self, runtime):
               import multiprocessing as mp
               from functools import partial
                
               files = {'thsamples':self.inputs.thsamples, 'fsamples':self.inputs.fsamples, 'phsamples':self.inputs.phsamples, 
                        'mask':self.inputs.mask, 'seed_ref':self.inputs.seed_ref, 'xfm':self.inputs.xfm}
               d = options | files
               
               ld = len(self.inputs.ds)
               pool = mp.Pool(ld)
               self.r = pool.map( partial(self.run_probtrack,d=d), range(ld))
               pool.close()
               
               return runtime
           
            def run_probtrack(self, n, d):
                from nipype.interfaces import fsl
                from nipype import Node
                import shutil
                
                d = d|self.inputs.ds[n]
                
                track_node = Node(fsl.ProbTrackX2(onewaycondition=True, step_length=0.5, mask=d['mask'],
                                dist_thresh=0.0, n_steps=d['n_steps'], c_thresh=0.2, args=d['args'],
                                thsamples=d['thsamples'], fsamples=d['fsamples'], phsamples=d['phsamples'],
                                loop_check=True, n_samples=d['n_samples'], omatrix1=True, xfm=d['xfm'],
                                meshspace="freesurfer", waycond='AND', mod_euler=True, seed_ref=d['seed_ref'],
                                seed=d['seed'], stop_mask=d['stop'], avoid_mp=d['avoid'], waypoints=d['waypoint'],
                                target_masks=d['target'], os2t=True),
                     name='tract_{}'.format(n))
            
                track_node.config = {'execution': {'stop_on_first_crash': False,
                                                   'hash_method': 'content',
                                                   'remove_unnecessary_outputs': False,
                                                   'crashfile_format': 'txt',
                                                   'crashdump_dir': d['crashdir']}}
                        
                track_node.base_dir = os.getcwd()
                result = track_node.run()
                
                tr_dir = os.path.join(track_node.base_dir, track_node.name)
                r = result.outputs.get()
                if ('--savepaths' in track_node.interface.cmdline) and (os.path.join(tr_dir,'saved_paths.txt')):
                    r = r | {'savepaths': os.path.join(tr_dir,'saved_paths.txt')}
                    
                if ('--savepaths' in track_node.interface.cmdline) and (os.path.join(tr_dir,'saved_paths.txt')):
                    r = r | {'savepaths': os.path.join(tr_dir,'saved_paths.txt')}
        
                return r
            
            def _list_outputs(self):
                
                outputs = self._outputs().get()
                outputs["result"] = self.r
                
                return outputs
        
        n_samples=5000; n_steps=2000
        options = {'crashdir':crashdir, 'n_samples': n_samples, 'n_steps':n_steps, 
                   'args':'--wayorder --fibthresh=0.01 --sampvox=0.0'}             
        probtrackFS.always_run = False
        probtrack_fs = Node(probtrackFS(options=options), mem_gb=250, name='probtrack_fs')
        
        def get_waytotals(probtrack_outd):
            import os, numpy as np
            from misc_funs import extractBetween
            
            with open(probtrack_outd['log'], 'r') as f:
                logf = f.readlines()[0].strip()
            
            seedf = extractBetween(logf, ['--seed=',' '])[0]
            with open(seedf, 'r') as f:
                seedf = f.readlines()[0].strip()
            seed, seedh = extractBetween(os.path.basename(seedf), ['roi-','_'],['hemi-','_'])
            
            stopf = extractBetween(logf, ['--stop=',' '])[0]
            with open(stopf, 'r') as f:
                stopf = f.readlines()[-1].strip()
            target, targeth = extractBetween(os.path.basename(stopf), ['roi-','_'],['hemi-','_'])
            
            sub = extractBetween(os.path.basename(stopf), ['sub-','_'])[0]
            
            waytotal = int(np.loadtxt(probtrack_outd['way_total']))
            
            d = {'participant':'sub-'+sub, 'hemi_seed':seedh, 'roi_seed':seed, 'seed':'_'.join([seedh,seed]), 
                 'hemi_target':targeth, 'roi_target':target, 'target':'_'.join([targeth,target]), 
                 'streamlines':waytotal}
                        
            return d
        
        waytotals = MapNode(Function(function=get_waytotals, input_names=['probtrack_outd'],
                                     output_names=['d']), iterfield=['probtrack_outd'], name='waytotals')
        
        def unique_streamlines(orig, probtrack_results):
            import nibabel as nib, numpy as np, pandas as pd
            import re, os
            from misc_funs import extractBetween
        
            im = nib.load(orig)
            new_im = np.zeros(im.shape)
        
            fdt_paths = nib.load(probtrack_results['fdt_paths'])
            paths = fdt_paths.get_fdata()
        
            l = []
        
            tract_id = 0
            with open(probtrack_results['savepaths'], 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        tract = int(re.search('# (\d+)', line).group(1))
                        tract_id += 1
                    else:
                        coords = [round(float(s)) for s in line.split()]
                        l.append(coords+[tract]+[tract_id])
                    
            df = pd.DataFrame(l, columns=['x','y','z','tract','tract_id'])
        
            df_unique = df.loc[df['tract_id'].isin(df.groupby('tract_id').agg(tuple).drop_duplicates().index),]
        
            dfvalues = df_unique[['x','y','z']].apply(lambda x: paths[x[0],x[1],x[2]], axis=1)
            if not dfvalues.empty:
                df_unique['values'] = dfvalues
                
                new_im[tuple( df_unique[['x','y','z']].values.T)] = 1
                
                dti_tracts = nib.Nifti2Image(new_im, im.affine, im.header)
                nib.save(dti_tracts, os.path.abspath('check_fdt_paths.nii.gz'))
            else:
                df_unique['values'] = None
                
            filename = os.path.abspath('unique_valid_tracts.tsv')
            df_unique.to_csv(filename, sep='\t')
            
            with open(probtrack_results['log'], 'r') as f:
                logf = f.readlines()[0].strip()
            
            seedf = extractBetween(logf, ['--seed=',' '])[0]
            with open(seedf, 'r') as f:
                seedf = f.readlines()[0].strip()
            seed, seedh = extractBetween(os.path.basename(seedf), ['roi-','_'],['hemi-','_'])
            
            stopf = extractBetween(logf, ['--stop=',' '])[0]
            with open(stopf, 'r') as f:
                stopf = f.readlines()[-1].strip()
            target, targeth = extractBetween(os.path.basename(stopf), ['roi-','_'],['hemi-','_'])
            
            sub = extractBetween(os.path.basename(stopf), ['sub-','_'])[0]
                
            d = {'participant':'sub-'+sub, 'hemi_seed':seedh, 'roi_seed':seed, 'seed':'_'.join([seedh,seed]), 
                 'hemi_target':targeth, 'roi_target':target, 'target':'_'.join([targeth,target]), 
                 'streamlines':len(df.tract_id.unique())}
            
            return d, filename
            
        get_unique_tracts = MapNode(Function(function=unique_streamlines, input_names=['orig','probtrack_results'],
                                             output_names=['d','filename']), iterfield=['probtrack_results'],
                                    name='get_unique_tracts')
        
        def join_results(ds, DTI_group_dir, AG, exp, filename):
            import os, pandas as pd, numpy as np
            from itertools import chain
            
            df = pd.DataFrame(chain.from_iterable(ds))
        
            df[['AG','study']] = AG,exp
            
            df = df.sort_values(['AG','study','participant','seed','target'], ascending=True, ignore_index=True)
        
            outfile = os.path.join(DTI_group_dir, 'space-fs_desc-{}_df.tsv'.format(filename))
            df[['participant','AG','study','hemi_seed','roi_seed','seed',
                'hemi_target','roi_target','target','streamlines']].to_csv(outfile, sep='\t', index=False)
            
            return outfile
            
        streamlines_group_df = JoinNode(Function(function=join_results, input_names=['ds','DTI_group_dir','AG','exp','filename'], output_names=['outfile']),
                                        joinsource=infosource, joinfield=['ds'], nested=True, name='streamlines_group_df')
        streamlines_group_df.inputs.DTI_group_dir = group_DTI_dir
        streamlines_group_df.inputs.AG = AG
        streamlines_group_df.inputs.exp = exp
        streamlines_group_df.inputs.filename = 'streamlinesPaths'
        streamlines_group_df.overwrite = False
        
        streamlines_group_df2 = streamlines_group_df.clone('streamlines_group_df2')
        streamlines_group_df2.inputs.filename = 'streamlinesWaytotal'
        streamlines_group_df2.overwrite = False
        
        datasink = Node(DataSink(base_directory=DTI_dir), name="datasink")
        substitutions = [('_ptc_','')]
        datasink.inputs.substitutions=substitutions
        
        wf_DTI.connect([\
                        (infosource, select_files, [('ptc', 'ptc')]),
                        (infosource, get_fmriprep_files, [('ptc', 'ptc')]),
                         
                        #DENOISE & GIBBS CORRECTION ON CONCATENATED AP AND PA IMAGES
                        (select_files, joinfiles, [('dwi', 'in1'), ('oppPE', 'in2')]),              
                        (joinfiles, merge_AP_PA, [('out','in_files')]),
                        (merge_AP_PA, denoise, [('merged_file', 'in_file')]),
                        (denoise, gibbs, [('out_file', 'in_file')]),                      
                        (gibbs, get_data_after_gibbs, [('out_file', 'image')]),
                        (select_files, get_data_after_gibbs, [('dwi', 'dwi_raw_file')]),
                        
                        (select_files, acqparams_index_files, [('dwi', 'dwi_file'),
                                                                ('dwi_json', 'dwi_json'),
                                                                ('oppPE_json', 'opp_json')]),
                        (gibbs, acqparams_index_files, [('out_file','APPA')]),
                        
                        #TOPUP
                        (get_data_after_gibbs, topup, [('B0_file', 'in_file')]),
                        (acqparams_index_files, topup, [('acqparams_file', 'encoding_file')]),
                                         
                        (topup, mean_image, [('out_corrected','in_file')]),
                        (mean_image, bet, [('out_file','in_file')]),
                         
                        #EDDY
                        (get_data_after_gibbs, eddy, [('dwi_file', 'in_file')]),
                        (select_files, eddy, [('bvals', 'in_bval'),
                                              ('bvecs', 'in_bvec')]),
                        (acqparams_index_files, eddy, [('index_file', 'in_index'),
                                                        ('acqparams_file', 'in_acqp')]),
                        (bet, eddy, [('mask_file','in_mask')]),
                        (topup, eddy, [('out_fieldcoef','in_topup_fieldcoef'),
                                       ('out_movpar', 'in_topup_movpar')]),
                         
                    
                        #EDDY QUAD             
                        (bet, eddy_quad, [('mask_file','mask_file')]),
                        (select_files, eddy_quad, [('bvals', 'bval_file'),
                                                  ('bvecs', 'bvec_file')]),
                        (eddy, eddy_quad, [(('out_corrected', base_name), 'base_name')]),
                        
                        (acqparams_index_files, eddy_quad, [('index_file', 'idx_file'),
                                                            ('acqparams_file', 'param_file')]),
               
                        (eddy, bias_correct, [('out_corrected','in_file'),
                                              ('out_rotated_bvecs','in_bvec')]),
                        (select_files, bias_correct, [('bvals','in_bval')]),
                        (bet, bias_correct, [('mask_file','in_mask')]),
                                         
                          
                        #DTIFIT
                        (select_files, dtifit, [('bvals','bvals')]),
                        (eddy, dtifit, [('out_corrected','dwi'),
                                        ('out_rotated_bvecs','bvecs'),
                                        ]),
                        (bet, dtifit, [('mask_file','mask')]),
                     
                        # ALIGN FA TO FREESURFER ORIG
                        (infosource, bbreg, [('ptc','subject_id')]),
                        (dtifit, bbreg, [('FA', 'source_file')]),
                        
                        # INVERT BBREG REG BECAUSE IT'S FROM FA (SRC) TO FS, AND WE WANT FS -> FA
                        # useful if you want to track from seeds in freesurfer's conformed space
                        (bbreg, FS2FA_mat, [('out_fsl_file','in_file')]),
                     
                        (get_fmriprep_files, mriconvert, [('orig', 'in_file')]),
                        
                        # convert the FS2T1 ANTs reg mat to FSL-type
                        # useful if you want to track from T1w seeds and need to transform labels in freesurfer space to T1w
                        (mriconvert, itk2fsl_FS2T1w_reg, [('out_file','source_file')]),
                        (get_fmriprep_files, itk2fsl_FS2T1w_reg, [('fs2T1w_reg', 'itk_file'),
                                                                  ('T1w', 'reference_file')]),  
                        (get_fmriprep_files, FS2anat_tk, [('orig','moving_image'),
                                                          ('T1w','target_image')]),
                        
                        (infosource, labels2T1w, [('ptc', 'subject_id')]),
                        (FS2anat_tk, labels2T1w, [( ('reg_file',arg_reg), 'args')]),
                        (get_fmriprep_files, labels2T1w_args, [('p_freesurfer', 'fs_dir')]),
                        (labels2T1w_args, labels2T1w, [('srclabel','source_label'),
                                                       ('trglabel','out_file')]),
                                    
                        # CONCAT THE REGISTRATION MATS: FA->FS AND FS->T1W
                        (bbreg, FA2anat_mat, [('out_fsl_file','in_file')]),
                        (itk2fsl_FS2T1w_reg, FA2anat_mat, [('out_reg','in_file2')]),
        
                        # GET ANAT2FA (i.e., INVERT FA2anat)
                        (FA2anat_mat, anat2FA_mat, [('out_file','in_file')]),
                         
                        # DISASSEMBLE H5 FILES
                        (get_fmriprep_files, disassemble_h5_T1w2MNI, [('h5_from_T1w_to_MNI152NLin6Asym','in_file')]),
                        (get_fmriprep_files, disassemble_h5_MNI2T1w, [('h5_from_MNI152NLin6Asym_to_T1w','in_file')]),
                       
                        # GET AFFINE FROM T1w TO MNI, AND MAKE IT FSL-COMPATIBLE
                        (disassemble_h5_T1w2MNI, itk2fsl_T1w2MNI_reg, [('affine_transform', 'itk_file')]),
                        (get_fmriprep_files, itk2fsl_T1w2MNI_reg, [('T1w', 'source_file')]),
                        
                        # REGISTER FA TO MNI (combine FA2ANAT and ANAT2MNI)
                        (FA2anat_mat, FA2MNI_mat, [('out_file','in_file')]),
                        (itk2fsl_T1w2MNI_reg, FA2MNI_mat, [('out_reg','in_file2')]),
                        
                        (dtifit, FA2anat_regcheck, [('FA','in_file')]),                            
                        (FA2anat_mat, FA2anat_regcheck, [('out_file','in_matrix_file')]),
                        (get_fmriprep_files, FA2anat_regcheck, [('T1w','reference')]),
                               
                        (disassemble_h5_T1w2MNI, convert_warp, [('displacement_field','from_itk')]),
                        (get_fmriprep_files, convert_warp, [('T1w','src_file')]),
                        
                        (convert_warp, FA2MNI_warp, [('out_file','warp1')]),
                        (FA2MNI_mat, FA2MNI_warp, [('out_file','premat')]),
                                      
                        #BEDPOSTX
                        (eddy, bedpostx, [('out_corrected','dwi'),
                                          ('out_rotated_bvecs','bvecs')]),
                        (select_files, bedpostx,[('bvals','bvals')]),
                        (bet, bedpostx, [('mask_file', 'mask')]),
                                        
                        #get labels
                        (infosource, ROIs_seed_fs, [('ptc','ptc')]),
                        (infosource, ROIs_seed_anat, [('ptc','ptc')]),                
        
                        # PROBTRACKX FREESURFER SEEDS  
                        (infosource, files_tract, [('ptc', 'ptc')]),
                        (bet, probtrack_fs, [('mask_file','mask')]),
                        (bedpostx, probtrack_fs, [('merged_thsamples','thsamples'),
                                              ('merged_fsamples','fsamples'),
                                              ('merged_phsamples','phsamples')]),
                        # seeds are in freesurfer conformed space
                        (FS2FA_mat, probtrack_fs, [('out_file','xfm')]),
                        (files_tract, probtrack_fs, [('d', 'ds')]),
                        (select_files, probtrack_fs, [('orig', 'seed_ref')]),
                        
                        (probtrack_fs, waytotals, [('result','probtrack_outd')]),
                        
                        (select_files, get_unique_tracts, [('orig', 'orig')]),
                        (probtrack_fs, get_unique_tracts, [('result', 'probtrack_results')]),
                        
                        (get_unique_tracts, streamlines_group_df, [('d','ds')]),
                        (waytotals, streamlines_group_df2, [('d','ds')]),
                                        
                         ])

        res_DTI = wf_DTI.run('MultiProc')
