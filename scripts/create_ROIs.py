
###########################
# NEED TO HAVE RUN:
#   F02_pipeline_BIDS
#   F02_pipeline_fmriprep

# For cerebellum ROIS:
#   F02_pipeline_SUIT (rerun after running this pipeline)
###########################

import os, glob, errno, re
import numpy as np, pandas as pd, nibabel as nib, nilearn as nil
from nipype import Workflow,Node,Function,MapNode,IdentityInterface,config
from nipype.interfaces.io import SelectFiles,DataSink
from subprocess import call
from nipype.interfaces import fsl, ants
from nipype.interfaces.base import isdefined,TraitedSpec,CommandLineInputSpec,CommandLine,File,OutputMultiPath,traits,Directory
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.freesurfer import MRIsConvert,MRIConvert,BBRegister,Label2Vol,Binarize
from nipype.interfaces.fsl.base import FSLCommandInputSpec, FSLCommand
from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec

AG_EXP = {'A02':['Extinction_EEG_fMRI'], 'A03':['3T','3T_EEG'], 'A05':['study3'],
          'A08':['ATOACQ','ContextII','Cort','TIA'], 
          'A09':['Extinction_Generalization_I','Extinction_Generalization_II'],
          'A12':['study_1','study_2']}

desc_name = 'ReducedAdd'

for AG, studies in AG_EXP.items():
    for exp in studies:
        
        bids_dir = os.path.join('/media/f02/F02',AG,exp)
        
        wf_dir = os.path.join(bids_dir, 'workflows')
        source_dir = os.path.join(bids_dir, 'sourcedata')
        rawdata_dir = os.path.join(bids_dir, 'rawdata')
        derivatives_dir = os.path.join(bids_dir, 'derivatives')
        func_denoised_dir = os.path.join(derivatives_dir, 'func_denoised')
        ROIs_dir = os.path.join(derivatives_dir, 'ROIs')
        freesurfer_dir = os.path.join(derivatives_dir, 'freesurfer')
        fmriprep_dir = os.path.join(derivatives_dir, 'fmriprep')
        crashdir = os.path.join(bids_dir, 'crashfiles')
        
            
        if not os.path.exists(ROIs_dir): os.makedirs(ROIs_dir)
        
        selected_rois = {
                'Bilateral_Hippocampus':
                    {'Left_Hippocampus':['Left-Hippocampus'],
                     'Right_Hippocampus':['Right-Hippocampus']},
                'Bilateral_Amygdala':
                    {'Left_Amygdala':['Left-Amygdala'],
                     'Right_Amygdala':['Right-Amygdala']},
                'Bilateral_vmPFC':
                    {'Left_vmPFC':['ctx-lh-medialorbitofrontal'], 
                     'Right_vmPFC':['ctx-rh-medialorbitofrontal']}, 
                'Bilateral_dACC':
                    {'Left_dACC':['ctx-lh-caudalanteriorcingulate'],
                     'Right_dACC':['ctx-rh-caudalanteriorcingulate']},
				'Bilateral_Thalamus':
                    {'Left_Thalamus':['Left-Thalamus'],
                     'Right_Thalamus':['Right-Thalamus']},
                'Both_LateralVentricle': 
                    {'Both_LateralVentricle':['Right-Lateral-Ventricle', 'Left-Lateral-Ventricle']},
                'Mid_3rdVentricle':
                    {'Mid_3rdVentricle':['3rd-Ventricle']},
                'Mid_4thVentricle':
                    {'Mid_4thVentricle':['4th-Ventricle']},
                    }
                    
        fs_file = pd.read_table(os.path.join(os.getenv('FREESURFER_HOME'),'FreeSurferColorLUT.txt'), sep=r'\s+',
                                comment='#', header=None, names=['ori_index','region','R','G','B','A'], engine='python')
        FS_dict = dict(fs_file[['region','ori_index']].to_dict(orient='split')['data'])
        df1 = pd.DataFrame(selected_rois).unstack().dropna()
        df2 = pd.concat([df1, df1.apply(lambda x: (fs_file[fs_file['region'].isin(x)].ori_index).tolist())],axis=1)
        df2['my_index'] = [[i] for i in range(1,df2.shape[0]+1)] 
        df_all = pd.concat([df2.groupby(level=0).sum(), df2.groupby(level=1).sum()]).astype(str).drop_duplicates()
        df_final = df_all.reset_index().rename(columns={'index':'region',0:'ori_label',1:'ori_index'})      
        df_final[['hemisphere','region']] = df_final['region'].str.split(pat='_', n=1, expand=True) 
        df_final['desc'] = 'aparcaseg'
        
        tissues = {'WM':  ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter'], 
                   'CSF': ['Left-Lateral-Ventricle','Left-Inf-Lat-Vent','3rd-Ventricle','4th-Ventricle','CSF',
                           'Right-Lateral-Ventricle','Right-Inf-Lat-Vent','5th-Ventricle'] }
        
        ptc_tsv = pd.read_csv(os.path.join(rawdata_dir, 'participants.tsv'), 
                              sep='\t', usecols=['participant_id','rsfMRI','T1w'])
        participants = ptc_tsv.loc[ptc_tsv.T1w==1,'participant_id'].tolist()
        
        name_wf_ROIs = 'wf_create_ROIs'
        wf_create_ROIs = Workflow(name=name_wf_ROIs, base_dir=wf_dir)
        wf_create_ROIs.config['execution'] = {'use_relative_paths': False,
                                             'stop_on_first_rerun': True,
                                             'hash_method': 'content',
                                             'crashfile_format': 'txt',
                                             'crashdump_dir': crashdir}
        
        df_final_file = os.path.join(ROIs_dir,'desc-ROIs_df.tsv')
        if not os.path.isfile(df_final_file):
            df_final.to_csv(df_final_file, sep='\t')
        else:
            df_old = pd.read_csv(df_final_file, sep='\t')
            if not df_old.equals(df_final):
                df_final.to_csv(df_final_file, sep='\t')
                
                
        # =============================================================================
        # INFOSOURCE - A FUNCTION FREE NODE TO ITERATE OVER THE LIST OF SUBJECT NAMES
        # =============================================================================
        infosource = Node(IdentityInterface(fields=['ptc']), name="infosource")
        infosource.iterables = [('ptc', participants)]
        
        templates = {'T1w': os.path.join(fmriprep_dir,'{ptc}','anat','{ptc}_desc-preproc_T1w.nii.gz'),
                     'aparc_aseg_fs': os.path.join(freesurfer_dir,'{ptc}','mri','aparc+aseg.mgz'),
                     'annot_aparc': os.path.join(freesurfer_dir,'{ptc}','label','*aparc.annot'),
                     'white_surf': os.path.join(freesurfer_dir,'{ptc}', 'surf', '[rl]h.white'),
                     'orig': os.path.join(freesurfer_dir,'{ptc}', 'mri', 'orig.mgz'),
                     'pials': os.path.join(freesurfer_dir,'{ptc}', 'surf', '[lr]h.pial'),
                     'anat2FS_mat': os.path.join(fmriprep_dir,'{ptc}', 'anat', '*from-T1w_to-fsnative_*xfm.txt')}
        
        select_files = Node(SelectFiles(templates, raise_on_empty=False), name='select_files')
        
        def all_atlases(ptc, derivatives_dir, pat):
            import os, glob
            p_fmriprep = os.path.join(derivatives_dir,'fmriprep',ptc)
            p_atlases = os.path.join(derivatives_dir,'atlases',ptc)
            aparc_aseg = glob.glob(os.path.join(p_fmriprep,'**/{}'.format(pat)),recursive=True)
            other_atlases = glob.glob(os.path.join(p_atlases,'**/*_atlas.nii.gz'),recursive=True)
        
            return aparc_aseg + other_atlases
        
        get_atlases = Node(Function(function=all_atlases, input_names=['ptc','derivatives_dir','pat'],
                                    output_names=['atlas_file']), name='get_atlases')
        get_atlases.inputs.derivatives_dir = derivatives_dir
        get_atlases.inputs.pat = '*desc-aparcaseg_dseg.nii.gz'
        
        
        def create_path_ROI(ptc,ROIs_dir):
            import os, errno
            p_rois = os.path.join(ROIs_dir, ptc)
            
            for subf in ['func','anat','MNI','dwi','freesurfer']:
                try:
                    os.makedirs(os.path.join(p_rois, subf))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
                    pass
        
            return p_rois
        
        ptc_ROI_path = Node(Function(function=create_path_ROI, input_names=['ptc','ROIs_dir'],
                       output_names=['p_rois']), name='ptc_ROI_path')
        ptc_ROI_path.inputs.ROIs_dir = ROIs_dir
            
        
        gunzip_anat = Node(Gunzip(), name='gunzip_anat')
        gunzip_boldref = Node(Gunzip(), name='gunzip_boldref')
        
        # # =============================================================================
        # # ACPC ALIGNMENT - T1 - run if necessary
        # # =============================================================================
        
        # def ACPCTask(T1_img, args=[]):
        #     from subprocess import Popen,PIPE,call
        #     import os, glob, shutil
        #     from misc_funs import split_filename
            
        #     [_,filename,ext] = split_filename(T1_img)
        #     new_file = os.path.abspath(filename+ext)
        #     shutil.copyfile(T1_img, new_file)
        
        #     cmd = 'acpcdetect -i {infile} {args}'.format(infile=new_file,
        #                                                  args=' '.join(args))
        #     Popen(cmd, shell=True,stdout=PIPE,cwd=os.getcwd()).stdout.read()
                
        #     fsl_mat = glob.glob(os.path.abspath('*_FSL.mat'))
            
        #     try:
        #         outori = [s.split()[1] for s in args if '-output-orient' in s][0]
        #     except IndexError:
        #         outori = 'RAS'
        #     out_file = glob.glob(os.path.abspath('*_{}.nii'.format(outori)))[0]
            
        #     with open(os.path.abspath('ooo.txt'),'w') as cmd_file:
        #         cmd_file.write(cmd)
                
        #     return out_file, fsl_mat
                
        # acpcdetect = Node(Function(function=ACPCTask, input_names=['T1_img','args'], 
        #                   output_names=['out_file','fsl_mat']), 
        #                   name='acpcdetect2')
        # acpcdetect.inputs.args = ['-center-AC', '-no-tilt-correction', '-output-orient RAS']
        
        def create_ROIs_atlas(ptc, df_rois, atlas_file, rois_to_atlas, p_rois, desc_name, erode_masks):
            import os, json, re
            from ast import literal_eval
            import numpy as np, nibabel as nib, pandas as pd
            from scipy import ndimage
            
            try:
                space =  re.match(r'.*_space\-([^\_]+)',os.path.basename(atlas_file)).group(1)
                if space.startswith('T1'):
                    subf = 'func'
                elif space.startswith('MNI'):
                    subf = 'MNI'
            except AttributeError:
                space = 'T1w'
                subf = 'anat'
                
            desc =  re.match(r'.*_desc\-([^\_]+)',os.path.basename(atlas_file)).group(1) 
        
            # =============================================================================
            # create reduced_atlas & freesurfer ROIs
            # =============================================================================
            # atlases
            atlas=nib.load(atlas_file)
            new_atlas_single = np.zeros(atlas.shape,dtype=int)
            
            labels = {}
            rois = []
            
            df_rois = pd.read_csv(df_rois, sep='\t', converters={'ori_label':literal_eval, 'ori_index':literal_eval, 'my_index':literal_eval})
            for nroi, row in df_rois.iterrows():
                idx = np.isin(atlas.get_fdata(), row['ori_index'])
                
                if row['region'] in erode_masks:
                    idx = ndimage.binary_erosion(idx)
                
                if not idx.any(): continue
            
            
                # save ROIs
                parc_save = nib.Nifti1Image(idx.astype(int), atlas.affine, atlas.header)    
                filename_parc_fs = '{sub}_space-{space}_roi-{region}_hemi-{hemi}_desc-{desc}_mask.nii.gz'.format(sub=ptc,
                                                                                                                   space=space,
                                                                                                                   region=row['region'],
                                                                                                                   hemi=row['hemisphere'],
                                                                                                                   desc=desc)
                roi_filename = os.path.join(p_rois,subf,filename_parc_fs)
                rois.append(roi_filename)
                parc_save.to_filename(roi_filename)
                        
                # create reduced atlas
                if (row.hemisphere!='Bilateral') and (row.region in rois_to_atlas):
                    
                    new_atlas_single[idx] = row['my_index']
                    
                    labels.update({'_'.join([row['hemisphere'],row['region']]):row['my_index'][0]})
                    labels = dict(sorted(labels.items(), key=lambda item: item[1]))
                    
                    # save reduced atlas    
                    reduced_atlas_single = nib.Nifti1Image(new_atlas_single, atlas.affine, atlas.header)
                    atlas_filename = '{sub}_space-{space}_desc-{desc}{n}_dseg.nii.gz'.format(sub=ptc,
                                                                                             space=space,
                                                                                             desc=desc,
                                                                                             n=desc_name)
                    reduced_atlas_filename = os.path.join(p_rois, subf, atlas_filename)
                    reduced_atlas_single.to_filename(reduced_atlas_filename)
                
                    labels_filename = '{sub}_space-{space}_desc-{desc}{n}_dseg.json'.format(sub=ptc,
                                                                                            space=space,
                                                                                            desc=desc,
                                                                                            n=desc_name)   
                    with open(os.path.join(p_rois, subf, labels_filename), 'w') as outfile:
                        json.dump(labels, outfile, indent=4)
                
            return rois, reduced_atlas_filename
        
        ROIs_and_reduced_atlas = MapNode(Function(function=create_ROIs_atlas, 
                                                  input_names=['ptc','df_rois','atlas_file','rois_to_atlas','p_rois','desc_name','erode_masks'],
                                                  output_names=['rois','reduced_atlas_filename']), 
                                         iterfield=['atlas_file'], name='ROIs_and_reduced_atlas')
        ROIs_and_reduced_atlas.inputs.df_rois = df_final_file
        ROIs_and_reduced_atlas.inputs.rois_to_atlas = ['Hippocampus','Amygdala','dACC','vmPFC',
                                                       'LateralVentricle','4thVentricle']
        ROIs_and_reduced_atlas.inputs.erode_masks = []
        ROIs_and_reduced_atlas.inputs.rois_to_atlas = ['Hippocampus','Amygdala','dACC','vmPFC',
                                                       'tTemporal','lOccipital','Pericalcarine','Supramarginal','Precentral','Lingual']
        ROIs_and_reduced_atlas.inputs.desc_name = desc_name
        
        
        # Create labels from annotation of ACC and vmPFC
        class Annotation2LabelInputSpec(FSTraitedSpec):
            annot_file =  File(exists=True, argstr='--annotation %s')
            subject_id = traits.Str(argstr='--subject %s', desc='subject id')
            hemi = traits.Enum('lh','rh', usedefault=True, argstr='--hemi %s', 
                               desc='hemisphere to use lh or rh', mandatory=True)
            out_dir = Directory(os.path.curdir, mandatory=True, argstr='--outdir %s', 
                                usedefault=True, desc='output directory')  
            subjects_dir = Directory(argstr='--sd %s')
            
        class Annotation2LabelOutputSpec(TraitedSpec):
            label_files = OutputMultiPath(File(exists=True))
            hemi = traits.Str()
        
        class Annotation2Label(FSCommand):
            _cmd = 'mri_annotation2label'
            input_spec = Annotation2LabelInputSpec
            output_spec = Annotation2LabelOutputSpec
            
            def _format_arg(self, name, spec, value):
                if name=='out_dir':
                    _,hemi,annot_name = re.split('^(rh|lh)\.',os.path.basename(self.inputs.annot_file))
        
                    self.outdir_labels = os.path.abspath(os.path.join(self.inputs.subjects_dir,
                                                                      self.inputs.subject_id,
                                                                      'labels',
                                                                      annot_name,
                                                                      hemi))
                    if not os.path.exists(self.outdir_labels): 
                        os.makedirs(self.outdir_labels)
                    return super(Annotation2Label,self)._format_arg(name,spec,self.outdir_labels)
                
                # when you call Annotation2Label you give it an annot_file which begins with either lh.*
                # or .rh*. You need to pass the hemisphere to mri_annotation2label.
                elif name=='hemi':
                    hemi='lh' if os.path.basename(self.inputs.annot_file).startswith('lh.') else 'rh'
                    self.inputs.hemi = hemi
                    return super(Annotation2Label,self)._format_arg(name,spec,hemi)
                
                else:
                    return super(Annotation2Label,self)._format_arg(name,spec,value)
                
            def _list_outputs(self):
                outputs = self.output_spec().get()
                outfile = glob.glob(os.path.join(self.outdir_labels,'**/*.label'),recursive=True)
                outputs['label_files'] = outfile
                outputs['hemi'] = self.inputs.hemi
                return outputs
        
        mri_annot2label_aparc = MapNode(Annotation2Label(subjects_dir=freesurfer_dir), iterfield=['annot_file'],
                                        name='mri_annot2label_aparc')
        
        mri_annot2label_aparc2009 = MapNode(Annotation2Label(subjects_dir=freesurfer_dir), iterfield=['args'],
                                        name='mri_annot2label_aparc2009')
        
        
        def get_fs_rois_args(ptc, df_final, ROIs_dir):
            import os
            import pandas as pd
            from ast import literal_eval
            
            df = pd.read_csv(df_final, sep='\t', converters={'ori_index':literal_eval},
                             usecols=['region','hemisphere','ori_index','desc'])
            df[['sub','space']] = ptc, 'fs'
            
            df['filename'] = df.loc[:,'sub'] +'_'+'space-'+df.loc[:,'space'] +'_'+ \
                'roi-'+df.loc[:,'region'] +'_'+ 'hemi-'+df.loc[:,'hemisphere'] +'_'+ \
                    'desc-'+df.loc[:,'desc'] + '_mask.nii.gz'
            
            match, filenames = df['ori_index'].to_list(), df['filename'].to_list()
            outfile = [os.path.join(ROIs_dir,ptc,'freesurfer',f) for f in filenames]
            
            return match, outfile
        
        fs_rois_args = Node(Function(function=get_fs_rois_args, input_names=['ptc','df_final','ROIs_dir'],
                                     output_names=['match','outfile']), name='fs_rois_args')
        fs_rois_args.inputs.df_final = df_final_file
        fs_rois_args.inputs.ROIs_dir = ROIs_dir
        
        extract_rois_aparc = MapNode(Binarize(out_type='nii.gz'), iterfield=['match','binary_file'], 
                                     name='extract_rois_aparc')
        
        
        # merge freesurfer labels of ROIs that have more than one freesurfer region (ACC and vmPFC)
        def merge_labels(ptc,p_rois,labels,hemi,which_labels,df):
            import shutil,os,re
            from misc_funs import flatten
            from ast import literal_eval
            import pandas as pd
        
            df_final = pd.read_csv(df, sep='\t', converters={'ori_label':literal_eval, 'ori_index':literal_eval, 'my_index':literal_eval})
            df = df_final[df_final.hemisphere!='Bilateral'] 
                
            lregions = which_labels.keys()
            out_files = []
            h = 'Left' if hemi=='lh' else 'Right'    
            df_rows = df[(df.region.isin(lregions))&(df.hemisphere==h)]
            for idx,row in df_rows.iterrows():
                rmatch = which_labels[row.region]
                r=re.compile('.*'+ '.*|.*'.join(rmatch)+ '.*')
                labels_match = list(filter(r.match,labels))
                l = '{ptc}_space-fs_roi-{region}_hemi-{hemi}_desc-{desc}_mask.label'.format(ptc=ptc,
                                                                                            region=row['region'],
                                                                                            hemi=row['hemisphere'],
                                                                                            desc=row['desc'])
                filename = os.path.join(p_rois,'freesurfer',l)
                vertices = []
                nlines = int() #keep track of nb of vertices and update the file
                with open(filename,'w') as outfile:  
                    for lfile in labels_match:
                        with open(lfile,'r') as label_file:
                            all_lines=label_file.readlines()
                            nlines+=int(all_lines[1])
                            vertices.extend(all_lines[2:])
                    outfile.write(all_lines[0])
                    outfile.write(str(nlines)+'\n')
                    vertices_str = ''.join(vertices)
                    outfile.write(vertices_str)
        
                out_files.append(filename)
                
                
            return out_files
        
        which_labels = {'dACC':['caudalanteriorcingulate'],
                        'vmPFC':['medialorbitofrontal']}
        merge_FS_labels = MapNode(Function(function=merge_labels, input_names=['ptc','p_rois','labels','hemi','which_labels','df'], 
                                     output_names=['out_files']), iterfield=['labels','hemi'],
                                  name='merge_FS_labels')
        merge_FS_labels.inputs.df = df_final_file 
        merge_FS_labels.inputs.which_labels = which_labels  
            
        # =============================================================================
        # convert freesurfer's labels with label2surf so that probtrackx2 can read them
        # =============================================================================
        
        convert_surf_gii = MapNode(MRIsConvert(out_datatype='gii'), iterfield=['in_file'], 
                                   name='convert_surf_gii')
            
        def get_labels(p_rois):
            import os, glob
            labels = glob.glob(os.path.join(p_rois, 'freesurfer', '*.label'))
            return labels
        
        
        def get_label_surf(label, surfs, p_rois):
            import os, glob
            from misc_funs import extractBetween, split_filename
            
            hemi = extractBetween(os.path.basename(label), ['hemi-','_'])
            h = 'lh' if hemi[0]=='Left' else 'rh'
            surf = [s for s in surfs if os.path.basename(s).startswith(h+'.')][0]
            filename = os.path.abspath(split_filename(label,'.label')[1])
            label_file = os.path.abspath('{}.txt'.format(filename))
            with open(label_file,'w') as outfile:
                outfile.write(label)
            
            return surf, label_file

        label_surf = MapNode(Function(function=get_label_surf, input_names=['surfs','label','p_rois'], 
                                      output_names=['surf','label_file']), iterfield=['label'],
                             name='label_surf')
                    
                    
        class Labels2SurfInputSpec(CommandLineInputSpec):
            surf = File(exists=True, argstr="-s %s", mandatory=True)
            label_file = File(argstr="-l %s")  
            out = File(name_source=['label_file'], argstr="-o %s", 
                       name_template='%s.gii', hash_files=False)
        
        class Labels2SurfOutputSpec(TraitedSpec):
            out = File(exists=True)
        
        class Labels2Surf(CommandLine):
            input_spec = Labels2SurfInputSpec
            output_spec = Labels2SurfOutputSpec
            _cmd = "label2surf"
             
        label2surf = MapNode(Labels2Surf(), iterfield=['surf','label_file'], 
                             name='label2surf')
        
        
        class Surf2VolumeInputSpec(CommandLineInputSpec):
            surf = File(exists=True, argstr="%s", mandatory=True, position=0)
            refvol = File(argstr="%s", position=1, mandatory=True)  
            outvol = File(name_source=['surf'], argstr="%s", position=2,
                          name_template='%s.nii.gz', hash_files=False)
            convention = traits.Str('freesurfer', argstr="%s", usedefault=True, 
                                    position=3, mandatory=True)
        
        class Surf2VolumeOutputSpec(TraitedSpec):
            outvol = File(exists=True)
        
        class Surf2Volume(CommandLine):
            input_spec = Surf2VolumeInputSpec
            output_spec = Surf2VolumeOutputSpec
            _cmd = "surf2volume"
            
            def _parse_inputs(self, skip=None):
                from misc_funs import split_filename
                skip=[]
                if isdefined(self.inputs.labels):
                    filename = '+'.join([split_filename(l, exts='.label')[1] for l in self.inputs.labels])
                    
                    labels_file = os.path.abspath('{}.txt'.format(filename))
                    with open(labels_file,'w') as outfile:
                        outfile.write('\n'.join(self.inputs.labels))
                    self.inputs.label_file = labels_file
                return super(Labels2Surf, self)._parse_inputs(skip=skip)
                    
            def _gen_fname(self, name):
                
                return
             
        surf2volume = MapNode(Surf2Volume(), iterfield=['surf'], name='surf2volume')
        
        mriconvert_orig = Node(MRIConvert(), name='mriconvert_orig')
        def mriconvert_outfile(filename):
            import os
            from misc_funs import split_filename
            [base,name,_] = split_filename(filename, exts='.mgz')
            return os.path.join(base, name+'.nii.gz')
        
        
        def get_tissues(ptc, aparc_aseg, fs_file, tissues, p_rois):
            import nibabel as nib, numpy as np, os
            return_files = []
            img = nib.load(aparc_aseg)
            data = img.get_fdata()
            ind_array = {k:fs_file.loc[fs_file['region'].isin(v)].ori_index.values for k,v in tissues.items()}
            for k,v in ind_array.items():
                tissue = np.isin(data,ind_array.get(k).astype(int))
                img_new = nib.nifti1.Nifti1Image(tissue, img.affine, header=img.header.copy())
         
                l = '{ptc}_space-fs_roi-{region}_hemi-Bilateral_desc-aparcaseg_mask.nii.gz'.format(ptc=ptc,
                                                                                                   region=k)
                filename = os.path.join(p_rois,'freesurfer',l)
        
                return_files.extend([filename])
                nib.save(img_new, filename) 
            return return_files
        
        extract_tissues = Node(Function(function=get_tissues, input_names=['ptc','aparc_aseg','fs_file','tissues','p_rois'], 
                                               output_names=['return_files']), name='extract_tissues')
        extract_tissues.inputs.tissues = tissues
        extract_tissues.inputs.fs_file = fs_file
        
        erode_imgs = MapNode(fsl.maths.ErodeImage(), iterfield=['in_file'], name='erode_img')
        
        # convert pials to .gii
        mriconvert_pials = MapNode(MRIsConvert(out_datatype='gii'), iterfield=['in_file'], 
                                   name='mriconvert_pials')
        
        def rename_pials(ptc, pials, p_rois):
            import os,shutil
            
            filenames = []
            for pial in pials:
                n = os.path.basename(pial)
                h = 'Left' if n[:2]=='lh' else 'Right'
                l = '{ptc}_space-fs_roi-pial_hemi-{h}_desc-surf_mask.gii'.format(ptc=ptc, h=h)
                        
                filename = os.path.join(p_rois,'freesurfer',l)                                                              
                shutil.copyfile(pial, filename)
                filenames.append(filename)  
                
            return filenames
        
        mv_pials = Node(Function(function=rename_pials, input_names=['ptc','pials','p_rois'], 
                                 output_names=['filenames']), name='mv_pials')
        
        
        def get_cerebellum(p_rois):
            from misc_funs import replaceValue
            import glob,os
            cerebellum_files = glob.glob(os.path.join(p_rois, 'anat', '*_roi-CerebellumNuclei_*.nii.gz'))
            cerebellum_filenames = [os.path.join(p_rois,'freesurfer',replaceValue(os.path.basename(f), ['space','fs']))
                                    for f in cerebellum_files]
            return cerebellum_files, cerebellum_filenames
        
        cerebellum_rois = Node(Function(function=get_cerebellum, input_names=['p_rois'], 
                                        output_names=['cerebellum_files','cerebellum_filenames']), 
                               name='cerebellum_rois')
        cerebellum_rois.overwrite = True
        
        cerebellum_fs = MapNode(ants.ApplyTransforms(interpolation='NearestNeighbor'), iterfield=['input_image','output_image'], 
                                name='cerebellum_fs')
        
                         
        datasink = Node(DataSink(parameterization=False), name="datasink")
        substitutions = [('_ptc_','')]
        datasink.inputs.substitutions=substitutions
            
        wf_create_ROIs.connect([\
                                (infosource, ptc_ROI_path, [('ptc','ptc')]),
                                (infosource, get_atlases, [('ptc','ptc')]),
                                (infosource, select_files, [('ptc','ptc')]),
                                
                                (select_files, gunzip_anat, [('T1w','in_file')]),
                                # (gunzip_anat, acpcdetect, [('out_file','T1_img')]),
        
                                (infosource, ROIs_and_reduced_atlas, [('ptc','ptc')]),
                                (ptc_ROI_path, ROIs_and_reduced_atlas, [('p_rois','p_rois')]),
                                (get_atlases, ROIs_and_reduced_atlas, [('atlas_file','atlas_file')]),
                                
                                (infosource, mri_annot2label_aparc, [('ptc','subject_id')]),
                                (select_files, mri_annot2label_aparc, [('annot_aparc','annot_file')]),
        
                                (infosource, merge_FS_labels, [('ptc','ptc')]),
                                (ptc_ROI_path, merge_FS_labels, [('p_rois','p_rois')]),
                                (mri_annot2label_aparc, merge_FS_labels,[('label_files','labels'),
                                                                         ('hemi', 'hemi')]),   
                                
                                (infosource, fs_rois_args, [('ptc','ptc')]),
                                (select_files, extract_rois_aparc, [('aparc_aseg_fs','in_file')]),
                                (fs_rois_args, extract_rois_aparc, [('match','match'),
                                                                    ('outfile','binary_file')]),
                                
                                (select_files, convert_surf_gii, [('white_surf','in_file')]),
                                (ptc_ROI_path, label_surf, [('p_rois', 'p_rois'),
                                                            ( ('p_rois',get_labels), 'label')]),
                                (convert_surf_gii, label_surf, [('converted', 'surfs')]),
                                
                                (label_surf, label2surf, [('label_file','label_file'),
                                                          ('surf', 'surf')]),   
                                
                                (label2surf, surf2volume, [('out', 'surf')]),
                                (select_files, mriconvert_orig, [('orig','in_file'),
                                                                 ( ('orig',mriconvert_outfile),'out_file')]),
                                
                                (mriconvert_orig, surf2volume, [('out_file', 'refvol')]),
                                
                                (infosource, extract_tissues, [('ptc', 'ptc')]),                        
                                (ptc_ROI_path, extract_tissues, [('p_rois', 'p_rois')]),
                                (select_files, extract_tissues, [('aparc_aseg_fs', 'aparc_aseg')]),
                                
                                (select_files, mriconvert_pials, [('pials', 'in_file')]),
                                (infosource, mv_pials, [('ptc', 'ptc')]),
                                (ptc_ROI_path, mv_pials, [('p_rois', 'p_rois')]),
                                (mriconvert_pials, mv_pials, [('converted', 'pials')]),
        
                                (ptc_ROI_path, cerebellum_rois, [('p_rois','p_rois')]),
                                (mriconvert_orig, cerebellum_fs, [('out_file','reference_image')]),
                                (select_files, cerebellum_fs, [('anat2FS_mat','transforms')]),
                                (cerebellum_rois, cerebellum_fs, [('cerebellum_files', 'input_image'),
                                                                  ('cerebellum_filenames', 'output_image')]),
                                
                                (ptc_ROI_path, datasink, [('p_rois','base_directory')]),
                                (label2surf, datasink, [('out','freesurfer.@f1')]),
                                ])
            
        wf_create_ROIs.run('MultiProc')
