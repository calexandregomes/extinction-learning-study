
import numpy as np, pandas as pd, os, glob, errno, shutil
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype import Workflow,Node,MapNode,Function,JoinNode,IdentityInterface, SelectFiles, DataSink
from pathlib import Path
import re
from subprocess import call

from nipype.interfaces.base import TraitedSpec,File,OutputMultiPath,traits,Directory,BaseInterface,BaseInterfaceInputSpec

# Specify directories and participant list
AG = 'A03'
exp = '3T'
AG_EDA = ['A02','A03','A05','A09','A12']
run_MRI = False
run_EDA = True
always_run_eda = True

base_dir = os.path.join('/media/f02/F02')
bids_dir = os.path.join(base_dir, AG, exp)
wf_dir = os.path.join(bids_dir, 'workflows')
rawdata_dir = os.path.join(bids_dir, 'rawdata')
source_dir = os.path.join(bids_dir, 'sourcedata')
derivatives_dir = os.path.join(bids_dir, 'derivatives')
dicom_dir = os.path.join(source_dir, 'dicoms')
crashdir = os.path.join(bids_dir, 'crashfiles')

for folder in [wf_dir, rawdata_dir, dicom_dir, derivatives_dir, crashdir]:
    try:
        os.makedirs(folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

ptcs_json = {"SFB1280_code":{"Description": "SFB1280 subject-specific code"},
             "participant_id": {"Description": "participant name in BIDS format"},            
             "participant_nb": {"Description": "number of participant as per experimentor"},            
             "participant_folder_name": {"Description": "this is the name of the participant-specific folder named by the experimenter"},            
             "AG": {"Description": "specific A group"},
             "rsfMRI": {"Description": "does it have resting-state fMRI data?","Levels":{0:"No",1:"Yes"}},
             "dwi": {"Description": "does it have diffusion data?","Levels":{0:"No",1:"Yes"}},
             "T1w": {"Description": "does it have a T1w scan?","Levels":{0:"No",1:"Yes"}},
             "EDA": {"Description": "which phases have EDA data?","Levels":{"AC":"Acquisition","EX":"Extinction","RE":"Renewal"}},
             "study": {"Description": "which study?"},
             "experimental_group":{"Description": "are there experimental groups (e.g., control vs placebo)?"},
             "genotype": {"Description": "genotype"},
             "age": {"Description": "age of the subject","Units":"years"},
             "sex": {"Description": "sex","Levels":{"M":"Male","F":"Female"}},
             "date_acquisition": {"Description": "date of acquisition"},
             "site": {"Description": "where were these data collected?"},
             "comments": {"Description": ""}}  


if AG in ['A02','A03','A08','A09','A12','A99']:
    p_ptc = Path(os.path.join(dicom_dir))
    participants = [f.name for f in p_ptc.iterdir() if f.is_dir()]
    participants_exp = [[exp]+s.split('_') for s in participants]
    
elif AG=='A05': # A05's data are already in nifti format
    participants_exp = pd.read_csv(os.path.join(dicom_dir,'participants.tsv'), delimiter='\t')
    participants_exp['study'], participants_exp['SFB1280_code'] = exp, participants_exp.participant_id.str.extract('sub-(.*)').astype(str)
    participants = participants_exp.participant_id.tolist()
    
elif AG=='A11':
    p_ptc = Path(os.path.join(dicom_dir))
    participants = [f.name for f in p_ptc.iterdir() if f.is_dir()]
    participants_exp = [[exp]+['{:0>3}'.format(s.split('sub-')[1])]+[s.split('sub-')[1]] 
                        for s in participants]

ptc_info_file = glob.glob(os.path.join(base_dir, AG,'*info.xlsx'))
if ptc_info_file:
    participants_info = pd.read_excel(ptc_info_file[0], sheet_name=exp,
                                      converters={'participant_nb':'{:0>3}'.format})


ptc_file_AG = os.path.join(rawdata_dir,'participants.tsv')
try:
    participants_file = pd.read_csv(ptc_file_AG, delimiter='\t', converters={'participant_nb':'{:0>3}'.format, 'SFB1280_code':str})
    miss_cols = [v for v in ptcs_json.keys() if not np.isin(v,participants_file.columns.values)]
    participants_file[miss_cols] = ''
except FileNotFoundError:
    participants_file = pd.DataFrame(columns=list(ptcs_json.keys()))

participants_list = pd.DataFrame()
if AG=='A02':
    participants_list['study'],participants_list['participant_folder_name'] = zip(*participants_exp)
elif AG in ['A03','A09','A11','A99']:
    participants_list['study'],participants_list['participant_nb'],participants_list['SFB1280_code'] = zip(*participants_exp)
elif AG=='A08':
    participants_list['study'],participants_list['SFB1280_code'] = zip(*participants_exp)
elif AG in ['A05']:
    participants_list = participants_exp[['study','SFB1280_code']]
elif AG=='A12':
    participants_list['study'],_,_,participants_list['SFB1280_code'] = zip(*participants_exp)
    
participants_list['AG'] = AG
participants_list['participant_folder_name'] = participants
    
if 'participants_info' in locals():
    participants_list = participants_list.merge(participants_info, how='outer')
    
# ensure types are correct, otherwise might get an error if e.g. a column is full of np.nan but you want to update it with strings    
participants_file = participants_file.astype({'experimental_group':str})

# Now combine the participants_file (which will become participants.tsv) with participants_list
common_cols = participants_file.columns.intersection(participants_list.columns).tolist()
participants_file_up = participants_file.merge(participants_list, how='outer', on=common_cols)
participants_file_up.participant_id = 'sub-'+participants_file_up['SFB1280_code']

# Check if new participants have been added
diff_cols = participants_list.columns.difference(participants_file.columns)
all_cols = participants_file_up.columns.union(diff_cols)

participants_exist = participants_file_up[participants_file_up.participant_folder_name.isin(participants)][all_cols.tolist()]

ptc_info_file2 = glob.glob(os.path.join(base_dir, AG,'*info2.xlsx'))
if ptc_info_file2:
    participants_info2 = pd.read_excel(ptc_info_file2[0], sheet_name=exp,
                                       converters={'participant_nb':'{:0>3}'.format})
    if not participants_info2.empty:
        participants_exist = (participants_info2.set_index('participant_id').combine_first(participants_exist.set_index('participant_id'))).reset_index()

if AG in ['A03']:
    participants_exist['experimental_group'].replace({'ABA':'renewal', 'ABB':'recall'}, inplace=True)
                                                      
order_cols = ['participant_id', 'SFB1280_code', 'AG', 'study','experimental_group', 
              'rsfMRI', 'dwi', 'T1w', 'EDA','participant_folder_name', 'participant_nb',
              'genotype', 'age', 'sex', 'date_acquisition','site', 'comments']
participants_exist[order_cols+diff_cols.tolist()].sort_values('participant_id').to_csv(ptc_file_AG, sep='\t',index=False)

# =============================================================================
# INFOSOURCE - A FUNCTION FREE NODE TO ITERATE OVER THE LIST OF SUBJECT NAMES
# =============================================================================

infosource = Node(IdentityInterface(fields=['ptc']), name="infosource")
infosource.iterables = [('ptc', participants)]

templates = {'dicom_dir': os.path.join(dicom_dir, '{ptc}')}
select_files_dicom = Node(SelectFiles(templates), name='select_files_dicom')

wf_create_bids = Workflow(name="wf_create_bids", base_dir=wf_dir)
wf_create_bids.config['execution'] = {'use_relative_paths': 'false',
                     'hash_method': 'content',
                     'stop_on_first_rerun': 'false',
                     'stop_on_first_crash': 'false',
                     'crashfile_format': 'txt',
                     'crashdump_dir': crashdir}

def CreateSourceSubDir(AG, sub, source_dir):
    import os, errno
    niftis_dir = os.path.join(source_dir, 'niftis', sub)
    physio_dir = os.path.join(source_dir, 'physio', sub)
    if AG in ['A05','A08','A09','A11']:
        beh_dir = os.path.join(source_dir, 'beh', sub)
    else:
        beh_dir = os.path.join(source_dir, 'beh')  

    for subdir in [niftis_dir,physio_dir,beh_dir]:
        try:
            os.makedirs(subdir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
                pass           
    return niftis_dir,physio_dir,beh_dir

outdir_source = Node(Function(function=CreateSourceSubDir, input_names=['AG', 'sub','source_dir'], 
                              output_names=['niftis_dir','physio_dir','beh_dir']), name='outdir_source')
outdir_source.inputs.source_dir = source_dir
outdir_source.inputs.AG = AG
outdir_source.overwrite=True

# =============================================================================
# CONVERT TO NIfTI WORKFLOW
# =============================================================================

if AG not in ['A05','A11']:    
    converter = Node(Dcm2niix(), name='converter')
else:
    def get_BIDS_files(output_dir, source_dir):
        import os, re, shutil, glob
        from pathlib import Path
        import nibabel as nib
        from misc_funs import split_filename
        
        suffixes = ['bold','dwi','T1w','epi']
        l = []
        p = Path(source_dir)
        
        for suff in suffixes:
            if suff=='bold': 
                formt = '*task-rest*_{suff}*'.format(suff=suff)
            elif suff=='epi': 
                formt = '*ses-2_dir-PA*_{suff}*'.format(suff=suff)
            else:
                formt = '*_{suff}*'.format(suff=suff)
                     
            l.extend(list(p.glob(os.path.join('**',formt))))
            
        files = [shutil.copyfile(f, os.path.join(output_dir,f.name)) for f in l]
         
        bids =  [f for f in files if re.match(r".*.json",f)]
        converted_files =  [f for f in files if re.match(r".*(.nii|.nii.gz)$",f)]
        for ix,f in enumerate(converted_files):
            [basedir,filename,ext] = split_filename(f)
            if ext=='.nii':
                im = nib.load(f)
                new_f = os.path.join(basedir, filename+'.nii.gz')
                im.to_filename(new_f)
                os.remove(f)
                converted_files =  glob.glob(os.path.join(basedir, '*.nii.gz'))

        bvals =  [f for f in files if re.match(r".*.bval",f)]
        bvecs =  [f for f in files if re.match(r".*.bvec",f)]
        
        return bids, converted_files, bvecs, bvals
    
    converter = Node(Function(function=get_BIDS_files, input_names=['output_dir','source_dir'], 
                               output_names=['bids','converted_files','bvecs','bvals']), name='converter')
    
datasink = Node(DataSink(base_directory=os.path.join(source_dir,'fmriprep')), name="datasink")

def createBIDS_files(ptc, AG, exp, bids, converted_files, dicom_dir, rawdata_dir,
                     participants_exist, excl_bids_file, bvals=[], bvecs=[]):

    from pathlib import Path
    import numpy as np, nibabel as nib, pandas as pd
    import os, json, shutil, pydicom, re, collections, glob
    from misc_funs import flatten
    
    has_rsfMRI,has_dwi,has_T1w = 0,0,0
    
    files = list(flatten([bids,converted_files,bvecs,bvals]))
                
    if AG in ['A02','A03','A08','A09', 'A99']:
        anat_pat = 'Anat'
        func_pat = 'Rest'
        dti_pat = 'DTI_60'
        dti_opp = 'DTI_opp'
    elif AG in ['A05','A11']:
        anat_pat = 'T1w'
        func_pat = 'rest'
        dti_pat = 'dwi'
        dti_opp = 'epi'
    elif AG=='A12':
        anat_pat = 'MPRAGE'
        func_pat = 'Resting'
        dti_pat = 'DTI_\d'
        dti_opp = 'DTI_opp'
        
    func_file = [f for f in files if re.match(r".*{}.*.nii.gz$".format(func_pat),f)]
    
    if func_file:
        func = nib.load(func_file[0])
        try:
            TR = func.header.get_zooms()[3]
            nb_slices = func.shape[2]
            nb_vols = func.shape[3]
            has_rsfMRI = 1
        except IndexError:
            pass        
    
    pats = [r".*{}.*gz".format(dti_pat), r".*{}.*gz".format(dti_opp)]
    has_dwi = int(all([any(re.match(rex,f) for f in files)
                            for rex in pats]))
     
    
    idxs_ptc = participants_exist.loc[:,'participant_folder_name']==ptc
    sub_name = participants_exist.loc[idxs_ptc,'participant_id'].iat[0]
    sfbcode = participants_exist.loc[idxs_ptc,'SFB1280_code'].iat[0]

    for file in files:
        
        pathfile = Path(file)
        
        if pathfile.suffix=='.json': ext = '.json'
        elif pathfile.suffix=='.bval': ext = '.bval'
        elif pathfile.suffix=='.bvec': ext = '.bvec'
        else: ext = '.nii.gz'
        
        if re.search(anat_pat, pathfile.stem) is not None: #if *Anat*.json
            filename = sub_name+'_T1w'+ext
            bidsfile = os.path.join(rawdata_dir, sub_name, 'anat', filename)
            if ext=='.nii.gz': has_T1w=1
        
        elif (re.search(func_pat, pathfile.stem) is not None) and has_rsfMRI:   
            if ext=='.json':
                with open(pathfile, 'r+') as json_file:
                    data = json.load(json_file)
                    try:
                        slice_timing = data['SliceTiming']
                    except KeyError:
                        slice_timing = np.linspace(0,TR-(TR/nb_slices),nb_slices).tolist()  
                    
                    data.update({"TaskName":"rest",
                                 "SliceTiming":slice_timing,
                                 "SliceEncodingDirection":"k",
                                 "NumberVolumes": nb_vols})
                    json_file.seek(0)
                    json.dump(data, json_file, indent=4)  
                    json_file.truncate()
            filename = sub_name+'_task-rest_bold'+ext
            bidsfile = os.path.join(rawdata_dir, sub_name, 'func', filename)
                           
        elif re.search(dti_pat, pathfile.stem) is not None:                               
            if ext=='.json':
                if AG in ['A03','A08','A99']:
                    IM_dcm = glob.glob(os.path.join(dicom_dir,'**/*DTI_60/**/IM*'),recursive=True)[0]
                    dicom_hdr = pydicom.dcmread(IM_dcm)
                elif AG=='A09':
                    Xs = glob.glob(os.path.join(dicom_dir,'**/XX*'),recursive=True)           
                    for X in Xs:
                        dicom_hdr = pydicom.dcmread(X)
                        if dicom_hdr[('0008','103E')].value == 'DTI_60_70sl_2mm':
                            break
                    
                with open(pathfile, 'r+') as json_file:
                    data = json.load(json_file) 
    
					TotalReadOutTime = data['TotalReadoutTime']
					PhaseEncodingDirection = data['PhaseEncodingDirection']
                    
                    if PhaseEncodingDirection=='j':
                        dwi_dir = 'PA'
                    elif PhaseEncodingDirection=='j-':
                        dwi_dir = 'AP' 

                    data.update({"PhaseEncodingDirection": PhaseEncodingDirection,
                                 "TotalReadoutTime": TotalReadOutTime})                                  
                    json_file.seek(0)
                    json.dump(data, json_file, indent=4)
                    json_file.truncate()       
   
            filename = sub_name+ '_dir-' + dwi_dir + '_dwi'+ext
            bidsfile = os.path.join(rawdata_dir, sub_name, 'dwi', filename)
       
        elif (re.search(dti_opp, pathfile.stem) is not None) & (ext not in ['.bval','.bvec']):
            if ext=='.json':
                if AG in ['A03','A99']:
                    IM_dcm = glob.glob(os.path.join(dicom_dir,'**/*DTI_60_oppPE/**/IM*'),recursive=True)[0]
                    dicom_hdr = pydicom.dcmread(IM_dcm)
                elif AG=='A08':
                    try:
                        IM_dcm = glob.glob(os.path.join(dicom_dir,'**/*short/**/IM*'),recursive=True)[0]
                    except IndexError: 
                        try: IM_dcm = glob.glob(os.path.join(dicom_dir,'**/*opp*/**/IM*'),recursive=True)[0]
                        except: raise 
                    dicom_hdr = pydicom.dcmread(IM_dcm)

                elif AG=='A09':
                    Xs = glob.glob(os.path.join(dicom_dir,'**/XX*'),recursive=True)           
                    for X in Xs:
                        dicom_hdr = pydicom.dcmread(X)
                        if dicom_hdr[('0008','103E')].value == 'DTI_oppPE':
                            break
                    
                with open(pathfile, 'r+') as json_file:
                    data = json.load(json_file) 

					TotalReadOutTime = data['TotalReadoutTime']
					PhaseEncodingDirection = data['PhaseEncodingDirection']

                    if PhaseEncodingDirection=='j':
                        dir_val,PEdwi = 'PA','AP'
                    elif PhaseEncodingDirection=='j-':
                        dir_val,PEdwi = 'AP','PA'

                    data.update({"PhaseEncodingDirection": PhaseEncodingDirection,
                                 "TotalReadoutTime": TotalReadOutTime,
                                 "IntendedFor": "dwi/{s}_dir-{d}_dwi.nii.gz".format(s=sub_name,
                                                                                    d=PEdwi)})
                    json_file.seek(0)
                    json.dump(data, json_file, indent=4)
                    json_file.truncate()
                    
            filename = sub_name+ '_dir-' + dir_val + '_epi' + ext
            bidsfile = os.path.join(rawdata_dir, sub_name, 'fmap', filename)

        else: continue
        
        Path(bidsfile).parent.mkdir(parents=True, exist_ok=True)    
        shutil.copyfile(pathfile,bidsfile)
        
    with open(excl_bids_file, 'r+') as json_file:
        excl_bids = json.load(json_file)
    try:
        x = excl_bids[AG][exp][ptc]
        if 'func' in x: has_rsfMRI=0
        if 'dwi' in x: has_dwi=0
        if 'anat' in x: has_T1w=0
    except KeyError:
        pass
    
    l = {sfbcode:{'rsfMRI':has_rsfMRI, 'dwi':has_dwi, 'T1w':has_T1w}}    
    
    return l

createBIDS = Node(Function(function=createBIDS_files, input_names=['ptc','AG','exp','bids','converted_files',
                                                                   'dicom_dir','rawdata_dir',
                                                                   'participants_exist','excl_bids_file',
                                                                   'bvals','bvecs'],
                            output_names=['l']), name='createBIDS')
createBIDS.inputs.AG = AG
createBIDS.inputs.exp = exp
createBIDS.inputs.participants_exist = participants_exist
createBIDS.inputs.dicom_dir = dicom_dir
createBIDS.inputs.rawdata_dir = rawdata_dir
createBIDS.inputs.excl_bids_file = os.path.join(base_dir, 'desc-excludeBIDS_info.json')


dataset_description = {"Name":"SFB1280_Resting_State",
                       "BIDSVersion":"1.0.2",
                       "Authors":["Gomes, Carlos Alexandre",
                                  "Penate, Javier Schneider",
                                  "Labrenz, Franziska",
                                  "Spisak, Tamas",
                                  "Quick, Harald",
                                  "Kumsta, Robert",
                                  "Timmann, Dagmar",
                                  "Axmacher, Nikolai"]}  

try:
    with open(os.path.join(dicom_dir,'readme.txt'), 'r') as infile:
        r = infile.read()
except FileNotFoundError:
    r = ''

def UpdatePtcFile(l,ptcs_json,rawdata_dir,participants_exist,dataset_description,readme):
    import os, json
    import pandas as pd
              
    with open(os.path.join(rawdata_dir,'dataset_description.json'), 'w') as dataset_desc_json:
        json.dump(dataset_description, dataset_desc_json, indent=4)
    if not os.path.isfile(os.path.join(rawdata_dir,'README')):
        with open(os.path.join(rawdata_dir,'README'), 'w') as readme_file:
            readme_file.write(readme)
            
    with open(os.path.join(rawdata_dir,'README'), 'r+') as readme_file:
        lines = readme_file.read()
        d_sub = {k:v for ele in l for k,v in ele.items()}
        for ptc in d_sub.keys():
            exist_files = ', '.join([k for k,v in d_sub[ptc].items() if not v])
            line = '\nParticipant {ptc} missing the following datasets: {exist}'.format(ptc=ptc,exist=exist_files)
            if (line not in lines) and not all(d_sub[ptc].values()) :
                readme_file.write(line) 
    with open(os.path.join(rawdata_dir,'participants.json'), 'w') as participants_json:
        json.dump(ptcs_json, participants_json, indent=4)
    
    D={}
    [D.update(d) for d in l]
    participants_file = os.path.join(rawdata_dir,'participants.tsv')    
    new_ptc_file = participants_exist.set_index('SFB1280_code')
    new_ptc_file.update(pd.DataFrame(D).T)
    new_ptc_file.to_csv(participants_file, sep='\t', index=True)
        
    return participants_file,participants_json.name,dataset_desc_json.name,readme_file.name

update_ptc_file = JoinNode(Function(function=UpdatePtcFile, 
                                    input_names=['l','ptcs_json','rawdata_dir',
                                                 'participants_exist','dataset_description',
                                                 'readme'], 
                                    output_names=['participants_file','participants_json','dataset_desc_json','readme_file']), 
                           joinsource=infosource,joinfield=['l'], 
                           name='final_ptc_file')
update_ptc_file.inputs.ptcs_json = ptcs_json
update_ptc_file.inputs.rawdata_dir = rawdata_dir
update_ptc_file.inputs.participants_exist = participants_exist
update_ptc_file.inputs.dataset_description = dataset_description
update_ptc_file.inputs.readme = r

    
if AG not in ['A05','A11']:
    wf_create_bids.connect([\
                            (infosource, select_files_dicom, [('ptc', 'ptc')]),
                            (infosource, outdir_source, [('ptc', 'sub')]),
                            (select_files_dicom, converter, [('dicom_dir', 'source_dir')]),
                            (outdir_source, converter, [('niftis_dir', 'output_dir')])
                            ])
else:
    wf_create_bids.connect([\
                            (infosource, outdir_source, [('ptc', 'sub')]),
                            (infosource, converter, [('ptc', 'sub')]),
                            
                            (outdir_source, converter, [('niftis_dir', 'output_dir')]),
                            ])

if run_MRI:
    wf_create_bids.connect([\
                            (infosource, select_files_dicom, [('ptc', 'ptc')]),
                            (infosource, outdir_source, [('ptc', 'sub')]),
                            (select_files_dicom, converter, [('dicom_dir', 'source_dir')]),
                            (outdir_source, converter, [('niftis_dir', 'output_dir')]),
                            
                            (infosource, createBIDS, [('ptc', 'ptc')]),
                            (converter, createBIDS, [('bids','bids'),
                                                     ('converted_files','converted_files'),
                                                     ('bvals','bvals'),
                                                     ('bvecs','bvecs')]),
                            (createBIDS, update_ptc_file, [('l','l')]),
                            ])
        
    
# =============================================================================
#                                   EDA - BIDS
# =============================================================================

class CreateBidsEDAInputSpec(BaseInterfaceInputSpec):
    ptc = traits.Str()
    participants_exist = traits.Any()
    physio_dir = Directory()
    rawdata_dir = Directory()
    dicom_dir = Directory()
    beh_dir = Directory()
    task = traits.Str()
    AG = traits.Str()
    exp = traits.Str()
    readme = File()

class CreateBidsEDAOutputSpec(TraitedSpec):
    bidsfile = traits.Either(File(), None)

class CreateBidsEDA(BaseInterface):
    input_spec = CreateBidsEDAInputSpec
    output_spec = CreateBidsEDAOutputSpec
  
    def _run_interface(self, runtime):

        import os, glob, json, shutil, gzip
        import numpy as np, pandas as pd
        from mne.io import read_raw_brainvision
        from mne import read_annotations
        
        rawdata_dir=self.inputs.rawdata_dir
        dicom_dir=self.inputs.dicom_dir
        physio_dir=self.inputs.physio_dir
        beh_dir=self.inputs.beh_dir
        AG=self.inputs.AG
        ptc=self.inputs.ptc
        exp=self.inputs.exp
        task=self.inputs.task
            
        idxs_ptc = self.inputs.participants_exist.loc[:,'participant_folder_name']==ptc
        sub_name = self.inputs.participants_exist.loc[idxs_ptc,'participant_id'].iat[0]
        
        if AG in ['A02','A03']:
            
            for ext in ['.eeg','.vhdr','.vmrk']:
                if AG=='A03':
                    p = os.path.join(dicom_dir, ptc, '04_eda/**/*{task}*{ext}'.format(task=task,ext=ext))
                elif AG=='A02':
                    p = os.path.join(source_dir, 'physio', ptc, '*{task}*{ext}'.format(task=task, ext=ext))                                                
                                                                                                                     
                try:
                    src_file = glob.glob(p, recursive=True)[0]
                
                    filename = os.path.basename(src_file)
                    dst_file = os.path.join(self.inputs.physio_dir,filename)
                    if AG=='A03': shutil.copyfile(src_file,dst_file)
                    if ext=='.vhdr': 
                        dst_vhdr = dst_file[:]
                    elif ext=='.vmrk':
                        dst_vmrk = dst_file[:]
                
                except IndexError:               
                    line = '\nParticipant {ptc} does not have valid eeg files for task: {task}'.format(ptc=ptc,task=task)
                    with open(self.inputs.readme, 'r+') as readme_file:
                        lines = readme_file.read()
                        if line not in lines:
                            readme_file.write(line)  
                        self._bidsfile = None
                    return
              
            try:
                raw_all = read_raw_brainvision(dst_vhdr, scale=1000000)
                
            except FileNotFoundError:
                
                filename_noext = os.path.splitext(filename)[0]
              
                with open(dst_vhdr) as f:
                    lines = f.readlines()    
               
                with open(dst_vhdr, 'w') as f:
                    for item in lines:
                        if 'DataFile=' in item:
                            item = 'DataFile=' + filename_noext + '.eeg\n'
                        elif 'MarkerFile=' in item:
                            item = 'MarkerFile=' + filename_noext + '.vmrk\n'
                        f.write("%s" % item)
                        
                raw_all = read_raw_brainvision(dst_vhdr, scale=1000000)
            
            if 'GSR_MR_100_LEFT' in raw_all.ch_names:
                raw_eda = raw_all.copy().pick('GSR_MR_100_LEFT')
            else:
                line = '\nParticipant {ptc} does not have a valid EDA channel for task: {task}'.format(ptc=ptc,task=task)
                with open(self.inputs.readme, 'r+') as readme_file:
                    lines = readme_file.read()
                    if line not in lines:
                        readme_file.write(line)  
                    self._bidsfile = None
                return           
                    
            annots = read_annotations(dst_vmrk).to_data_frame()
            onsets = annots[annots.description.str.contains('Stimulus')]['onset']
            end_recording = raw_eda.times[-1]
            
            if not onsets.empty:            
                exp_end = annots.iloc[-1].onset.timestamp()
                # sometimes there are less  EDA recordings than what the last Stimulus trigger indicates
                if end_recording < exp_end: exp_end = end_recording

                t_last_trial = 15
                if onsets.shape[0]>1:
                    if onsets.iloc[-1].timestamp()+t_last_trial > exp_end:
                        last_t = exp_end     
                    else:
                        last_t = onsets.iloc[-1].timestamp()+t_last_trial
                else:
                    last_t = exp_end 
                
                raw_trimmed = raw_eda.copy().crop(tmin=onsets.iloc[0].timestamp(), tmax=last_t)
                trigger_array = [1]+[0]*(raw_trimmed.times.shape[0]-1)
                
                EDA_data = np.array([raw_trimmed.get_data()[0], trigger_array]).T
                
            else:
                line = '\nParticipant {ptc} does not have "Stimulus" marker in .vmrk file for task: {task}'.format(ptc=ptc,
                                                                                                                   task=task)
                with open(self.inputs.readme, 'r+') as readme_file:
                    lines = readme_file.read()
                    if line not in lines:
                        readme_file.write(line)
                self._bidsfile = None
                return
                                         
            if AG=='A03':
                beh_events = glob.glob(os.path.join(dicom_dir,'task-*{task}*_events.tsv'.format(task=task)))
            elif AG=='A02':
                source_beh_dir = os.path.join(source_dir,'beh','bids',ptc)
                beh_events = glob.glob(os.path.join(source_beh_dir,'*task-*{task}*_events.tsv'.format(task=task)))
                
            if beh_events: 
                behdat = pd.read_csv(beh_events[0], sep='\t')
                if behdat.iloc[0].onset > 0:
                    behdat.onset = behdat.onset - behdat.iloc[0].onset
                    behdat.to_csv(beh_events[0], sep='\t',index=False)
                
            sfreq = raw_eda.info['sfreq']
                
        else:
            if AG=='A05':
                [ses,run] = task.split('_')                       
                filename = '{sub}*{ses}*task-fear*{run}*physio.tsv.gz'.format(sub=sub_name,
                                                                              ses=ses,
                                                                              run=run)
                
                jsonfilename = '{sub}*{ses}*task-fear*{run}*physio.json'.format(sub=sub_name,
                                                                                ses=ses,
                                                                                run=run)
                
                behfilename = '{sub}*{ses}*task-fear*{run}*events.tsv'.format(sub=sub_name,
                                                                              ses=ses,
                                                                              run=run)
            elif AG=='A09':
                filename = '{sub}*task-{task}*_physio.tsv.gz'.format(sub=sub_name,
                                                                     task=task)
                
                jsonfilename = '{sub}*task-{task}*physio.json'.format(sub=sub_name,
                                                                      task=task)
                
                behfilename = '{sub}*task-{task}*events.tsv'.format(sub=sub_name,
                                                                    task=task)
                
            elif AG=='A12':
                sub_nb = participants_exist.loc[idxs_ptc,'participant_nb'].iat[0]
                sub_onsets = participants_exist.loc[idxs_ptc,'Onsets'].iat[0]
                
                filename = '{sub}_{task}.txt'.format(sub=sub_nb,task=task)
                
                t='ACQ' if task=='A' else 'EXT'
                behfilename = '{on}_{t}.tsv'.format(on=sub_onsets, t=t)
                
                data = {"SamplingFrequency": 2000,
                        "StartTime": 0.0,
                        "Columns": ["skinconductance", "trigger"],
                        "Units" : "microsiemens"}
                    
            try:
                eda_file = glob.glob(os.path.join(physio_dir,'**',filename), recursive=True)[0]
            except IndexError:               
                line = '\nParticipant {ptc} does not have valid eda files for task: {task}'.format(ptc=ptc,task=task)
                with open(self.inputs.readme, 'r+') as readme_file:
                    lines = readme_file.read()
                    if line not in lines:
                        readme_file.write(line)  
                    self._bidsfile = None
                return
            
            if AG != 'A12': #A12 doesn't have a jsonfile
                eda_json = glob.glob(os.path.join(physio_dir,'**',jsonfilename), recursive=True)[0]
                with open(eda_json, 'r') as json_file:
                    data = json.load(json_file)
                
                with gzip.open(eda_file,'rt') as f:
                    if AG=='A09': 
                        df_physio = pd.read_csv(f, sep='\t', header=None, index_col=False).transpose()
                        if len(df_physio.columns) > 1: 
                            df_physio = df_physio[[0]]
                            line = '\nParticipant {ptc} task {task} has more than one EDA column'.format(ptc=ptc,task=task)
                            with open(self.inputs.readme, 'r+') as readme_file:
                                lines = readme_file.read()
                                if line not in lines:
                                    readme_file.write(line)  
                            
                        df_physio.columns = data['Columns']
                        df_physio['trigger'] = [1] + [0] * (len(df_physio)-1)
                        
                    elif AG=='A05':
                        df_physio = pd.read_csv(f, sep='\t', header=None, index_col=False, 
                                            names=data['Columns'])
            else:
                df_physio = pd.read_csv(eda_file, sep='\t', header=None, index_col=False, 
                                    names=data['Columns'])
                df_physio['trigger'] = [1] + [0] * (len(df_physio)-1)
                
            sfreq = data['SamplingFrequency']
                    
            discard_rec = np.abs(sfreq*data['StartTime'])
            
            EDA_data = df_physio.loc[discard_rec:, ['skinconductance','trigger']].rename(columns={'skinconductance':'EDA'})

            beh_events = glob.glob(os.path.join(beh_dir,'**',behfilename), recursive=True)

            
        if task in ['AC','ac','ses-1_run-2','acq','A','ses-01_task-extinctioneegfmri_run-01']:
            task = 'acquisition'
        elif task in ['EX','ex','ses-1_run-3','E','ses-01_task-extinctioneegfmri_run-02']:
            task = 'extinction'
        elif task in ['RE','re','ren']:
            task = 'renewal'
        elif task in ['ses-2_run-1','rec']:
            task = 'recall'
                        
        # save EDA data in BIDs format   
        p_rawdata_dir = os.path.join(rawdata_dir, sub_name, 'func')
        filename = '{sub}_task-{task}_physio.tsv.gz'.format(sub=sub_name,task=task)
        if not os.path.isdir(p_rawdata_dir): 
            os.makedirs(p_rawdata_dir, exist_ok=True)
        
        bidsfile = os.path.join(p_rawdata_dir, filename)
        np.savetxt(bidsfile, EDA_data, delimiter='\t')
        
        # create and save json file associated with physio data
        json_physio = {"SamplingFrequency": sfreq,
                       "StartTime": 0.0,
                       "Columns": ["EDA", "trigger"],
                       "Units" : "microsiemens"}
        
        filename = '{sub}_task-{task}_physio.json'.format(sub=sub_name,task=task)
        bidsfile = os.path.join(p_rawdata_dir, filename)
        with open(bidsfile, 'w') as outfile:
            json.dump(json_physio, outfile, indent=4)
        
        # save events file associated with physio data
        filename = '{sub}_task-{task}_events.tsv'.format(sub=sub_name,task=task)
        bidsfile = os.path.join(p_rawdata_dir, filename)
        if beh_events:
            shutil.copyfile(beh_events[0],bidsfile)
        else:
            line = '\nParticipant {ptc} does not have a valid logfile for task: {task}'.format(ptc=ptc,task=task)
            with open(self.inputs.readme, 'r+') as readme_file:
                lines = readme_file.read()
                if line not in lines:
                    readme_file.write(line)  
                self._bidsfile = None
             
        self._bidsfile = bidsfile              
            
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["bidsfile"] = self._bidsfile
        return outputs


if AG in AG_EDA and run_EDA:

    CreateBidsEDA.always_run = always_run_eda
    createBIDS_EDA = MapNode(CreateBidsEDA(AG=AG, exp=exp, dicom_dir=dicom_dir,
                                           rawdata_dir=rawdata_dir, participants_exist=participants_exist),
                             iterfield=['task'], name='createBIDS_EDA')
    
    if AG=='A02':
        tasks = ['ses-01_task-extinctioneegfmri_run-01','ses-01_task-extinctioneegfmri_run-02']
                 'ses-02_task-extinctioneegfmri_run-01','ses-02_task-extinctioneegfmri_run-02']
    elif AG=='A03':  
        tasks = ['AC','EX','RE','ren']  
    elif AG=='A09':
        tasks = ['acq','ex','rec']
    elif AG=='A05':
        tasks = ['ses-1_run-2','ses-1_run-3','ses-2_run-1']
    elif AG=='A12':
        tasks = ['A','E']
    
          
    createBIDS_EDA.inputs.task = tasks
    
    if not run_MRI:
        readmef = os.path.join(rawdata_dir, 'README')
        if not os.path.isfile(readmef): 
            with open(readmef, mode='a'): pass
        templates = {'readme_file': readmef}
        update_ptc_file = Node(SelectFiles(templates), name='update_ptc_file')        
        
    wf_create_bids.connect([\
                            (infosource, createBIDS_EDA, [('ptc','ptc')]),
                            (update_ptc_file, createBIDS_EDA, [('readme_file','readme')]),
                            (infosource, outdir_source, [('ptc', 'sub')]),
                            (outdir_source, createBIDS_EDA, [('physio_dir', 'physio_dir'),
                                                             ('beh_dir','beh_dir')])
                            ])
        
res = wf_create_bids.run('MultiProc')
