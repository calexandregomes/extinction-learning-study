
import pandas as pd, multiprocessing as mp
import subprocess,shlex
from subprocess import Popen,PIPE,call
import os,datetime,psutil,warnings
from misc_funs import split_equal

# Specify directories and participant list
AG = 'A03'
exp = '3T'

bids_dir = os.path.join('/media/f02/F02', AG, exp)
bids_dir = os.path.join('/media/sf_SFB1280_Share/F02/DATA', AG, exp)

wf_dir = os.path.join(bids_dir, 'workflows')
rawdata_dir = os.path.join(bids_dir, 'rawdata_other_rest')
derivatives_dir = os.path.join(bids_dir, 'derivatives')
output_dir = os.path.join(derivatives_dir,'fmriprep_other_rest')
fmriprep_work_dir = os.path.join(wf_dir, 'wf_fmriprep_other_rest')
freesurfer_dir = os.path.join(derivatives_dir, 'freesurfer')
if not os.path.exists(fmriprep_work_dir): os.makedirs(fmriprep_work_dir)

fs_license = os.path.join(os.getenv('FREESURFER_HOME'),'license.txt')
participants_file = pd.read_csv(os.path.join(rawdata_dir, 'participants.tsv'), 
                                sep='\t', usecols=['participant_id','rsfMRI','T1w'])
participants_file = participants_file.loc[participants_file.rsfMRI==1,['participant_id','rsfMRI','T1w']]

subs_rerun = ['']

n_cpus = 16
omp_nthreads = 8
 
# =============================================================================
# FMRIPREP
# =============================================================================

def fmriprep_cmd(**kwargs):
    user_id = kwargs.get('user_id')
    bids_dir = kwargs.get('bids_dir')
    output_dir = kwargs.get('output_dir')
    fs_license = kwargs.get('fs_license')
    work_dir = kwargs.get('work_dir')
    version = kwargs.get('version')
    subs = kwargs.get('subs')
    task = kwargs.get('task')
    
    command = 'docker run --rm -i \
				-u {user_id} \
				-v {bids_dir}:/data:ro \
				-v {output_dir}:/out \
				-v {fs_license}:/opt/freesurfer/license.txt:ro \
				-v {fs_subs_dir}:/opt/subjects \
				-v {work_dir}:/scratch -w /scratch \
				nipreps/fmriprep:{version} /data /out \
				participant --participant-label {subs} '.format(user_id=user_id,
																bids_dir=bids_dir,
																output_dir=output_dir,
																fs_license=fs_license,
																work_dir=work_dir,
																version=version,
																subs=subs,
																fs_subs_dir=freesurfer_dir)
 
    if 'ignore' in kwargs.keys(): 
        ignore = kwargs.get('ignore')
        command += ' --ignore {} '.format(ignore)
    if 'output_space' in kwargs.keys(): 
        output_space = kwargs.get('output_space')
        command += ' --output-space {} '.format(output_space)
    if 'cifti_output' in kwargs.keys(): 
        cifti_output = kwargs.get('cifti_output')
        command += ' --cifti-output {} '.format(cifti_output)
    if 'dummy_scans' in kwargs.keys():
        dummy_scans= kwargs.get('dummy_scans')
        command += ' --dummy-scans {} '.format(dummy_scans)
    if 'n_cpus' in kwargs.keys():
        n_cpus = kwargs.get('n_cpus')
        command += ' --n_cpus {} '.format(n_cpus)
    if 'omp_nthreads' in kwargs.keys(): 
        omp_nthreads = kwargs.get('omp_nthreads')
        command += ' --omp-nthreads {} '.format(omp_nthreads)
    if 'mem' in kwargs.keys():
        mem = kwargs.get('mem')
        command += ' --mem {} '.format(mem)
    if 'args' in kwargs.keys():
        args = kwargs.get('args')
        command += ' ' + '  '.join(args) + ' '
    if 'task' in kwargs.keys(): 
        task = kwargs.get('task')
        command += ' --task-id {} '.format(task)
        
    command += ' -w /scratch --fs-subjects-dir /opt/subjects'

    return command

user_id = os.getuid() 
version = '23.1.4'
ignore = ' '.join( ['fieldmaps'] )
output_space = ' '.join( ['T1w'] )
# cifti_output = ''
dummy_scans = 2
# task = ' '.join( ['acquisition'] )
args = '--fs-subjects-dir {}'.format(freesurfer_dir)
cmds_dir = os.path.join(fmriprep_work_dir, 'fmriprep_cmds')
if not os.path.exists(cmds_dir): os.makedirs(cmds_dir)
    

ignore_params_check = ['--n_cpus', '--omp-nthreads', '--mem',
                       fs_license, bids_dir, output_dir, fmriprep_work_dir]
def update_cmd(cmd, ignore_params=ignore_params_check):
    updated_cmd = [param for param in cmd
                   if not any(rm_param in param for rm_param in ignore_params)]
    return updated_cmd

def fmriprep_run(sub):
     
    sub_id = sub.get('participant_id')
        
    if sub.get('rsfMRI')==1:
        fmriprep_command = fmriprep_cmd(
                                    #required 
                                    user_id=user_id,
                                    bids_dir=rawdata_dir, 
                                    output_dir=output_dir, 
                                    subs='{chunk}', 
                                    fs_license=fs_license, 
                                    fs_dir=freesurfer_dir,
                                    work_dir=fmriprep_work_dir,
                                    version=version,
                                    
									#optional
                                    ignore=ignore,
                                    output_space=output_space, 
                                    omp_nthreads=omp_nthreads,
                                    n_cpus=n_cpus,
                                    # cifti_output=cifti_output,
                                    dummy_scans=dummy_scans, 
                                    # task=task,
                                    args=args,
                                    )   
    elif sub.get('T1w')==1:
        fmriprep_command = fmriprep_cmd(
                                #required
                                user_id=user_id,
                                bids_dir=rawdata_dir, 
                                output_dir=output_dir, 
                                subs='{chunk}', 
                                fs_license=fs_license, 
                                work_dir=fmriprep_work_dir,
                                version=version,
								
                                #optional
                                ignore=ignore, 
                                output_space=output_space, 
                                args=['--anat-only'])   
    else:
        warnings.warn('Subject {} have neither resting-state nor DTI data!'.format(sub_id))
        return

    current_cmd = set(fmriprep_command.format(chunk=sub_id).strip().split('  '))
    current_cmd = update_cmd(current_cmd)

    filename = os.path.join(cmds_dir, sub_id+'.txt')    
    mode = 'r+' if os.path.isfile(filename) else 'a+'
   
    with open(filename, mode) as f:
        lines = list(f)
        last_cmd = lines[-1] if lines else ''
        previous_cmd = set(last_cmd.strip().split('  '))
        previous_cmd = update_cmd(previous_cmd)
        change = previous_cmd!=current_cmd
        
        if (change) or sub_id in subs_rerun: 
            
            startT = datetime.datetime.now()
            print('''\r\r********************************************
                  Currently preprocessing {sub_id}...
                  ********************************************\r\r'''.format(sub_id=sub_id)) 
                        
            # Popen(fmriprep_command.format(chunk=sub_id), shell=True, 
            #       stdout=PIPE,cwd=os.getcwd()).stdout.read()
            # 
            # p = subprocess.run(fmriprep_command.format(chunk=sub_id).split(maxsplit=1))
            fcmd = shlex.split(fmriprep_command.format(chunk=sub_id))
            p = subprocess.run(fcmd)
            
            # stdoutdata, stderrdata = p.communicate()
            
            completionT = datetime.datetime.now() - startT
            # If successfully ran, write out the command to file
            if p.returncode==0:
                if lines: f.write('\r\r')
                f.write(datetime.datetime.now().strftime('%x %X\t') + 
                        'Completion time: ' + str(completionT) + '\r\r')
                f.write(fmriprep_command.format(chunk=sub_id)+'\n')
            
                print('''\r\r********************************************
                      Finished preprocessing {sub_id}
                      Completion time: {time}
                      ********************************************\r\r'''.format(sub_id=sub_id,time=completionT))           
            else:
                print('''\r\r********************************************
                      !!!!!!!! fmriprep FAILED on subject {} !!!!!!!!!!
                      ********************************************\r\r'''.format(sub_id))

    
        else:
            print("Subject {sub_id} already preprocessed.".format(sub_id=sub_id)) 
  

n_subs_per_chunk = 5
d = participants_file.to_dict('records')
chunks = split_equal(d, n_subs_per_chunk)

# Run fmriprep on first subject to initialise
fmriprep_run(sub1)

# Run fmriprep on remaining subjects
for chunk in chunks:
    pool = mp.Pool(mp.cpu_count())
    pool.map(fmriprep_run, chunk)
    pool.close()
