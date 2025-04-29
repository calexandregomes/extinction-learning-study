
clearvars;

warning('off', 'MATLAB:rmpath:DirNotFound');
% remove PsPM from path because it conflicts with SPM
rmpath(genpath('/media/sf_G_80tb/installers/Toolboxes/PsPM6.1/'));

% Initialise SPM
%--------------------------------------------------------------------------
addpath('/usr/local/MATLAB/TOOLBOXES/spm12/');
spm('Defaults','fMRI');
spm_jobman('initcfg');

% Define some variables common to all groups
%--------------------------------------------------------------------------
mri_space = 'T1w';
confound_type = 'Satterthwaite';

desc = 'trimmed2None'; %if there is no desc, use ''
name_model = ['ReducedMaxconn','_',desc];
name_folder = '5rois';
network = 0;

% because sometimes there is no desc, create desc2
if ~isempty(desc)
    desc2 = ['_desc-',desc];
else
    desc2 = '';
end

connections = {
    {'Left_Amygdala','Left_Hippocampus'}, {'Right_Amygdala','Right_Hippocampus'},...
    {'Left_Amygdala','Left_vmPFC'}, {'Right_Amygdala','Right_vmPFC'},...
    {'Left_Amygdala','Left_dACC'}, {'Right_Amygdala','Right_dACC'},...
    {'Left_Amygdala','Right_CerebellumNuclei'}, {'Right_Amygdala','Left_CerebellumNuclei'},...
    {'Left_Hippocampus','Left_vmPFC'}, {'Right_Hippocampus','Right_vmPFC'},...
    {'Left_Hippocampus','Left_dACC'}, {'Right_Hippocampus','Right_dACC'},...
    {'Left_Hippocampus','Right_CerebellumNuclei'}, {'Right_Hippocampus','Left_CerebellumNuclei'},...
    {'Left_vmPFC','Left_dACC'}, {'Right_vmPFC','Right_dACC'},...
    {'Left_vmPFC','Right_CerebellumNuclei'}, {'Right_vmPFC','Left_CerebellumNuclei'},...
    {'Left_dACC','Right_CerebellumNuclei'}, {'Right_dACC','Left_CerebellumNuclei'},...
   };


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CREATE GROUPS AND STUDIES STRUCT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

AG_EXP = struct('A02',{{'Extinction_EEG_fMRI'}}, 'A03',{{'3T','3T_EEG'}}, 'A05',{{'study3'}},...
    'A08',{{'ATOACQ','ContextII','Cort','TIA'}},...
    'A09',{{'Extinction_Generalization_I','Extinction_Generalization_II'}},...
    'A12',{{'study_1','study_2'}});
 
fn = fieldnames(AG_EXP);
for ag = 1:length(fn)
    AG = fn{ag};
    studies = AG_EXP.(AG);
    fprintf('\n\nAG: %s\n\n',AG);
    for ex = 1:length(studies)
        exp = studies{ex};
        fprintf('\n\nStudy: %s\n\n',exp);

        base_dir = '/media/f02/F02';
        bids_dir = fullfile(base_dir, AG, exp);
        rawdata_dir = fullfile(bids_dir, 'rawdata');
        derivatives_dir = fullfile(bids_dir, 'derivatives');
        fmriprep_dir = fullfile(derivatives_dir, 'fmriprep');
        denoised_dir = fullfile(derivatives_dir, 'func_denoised');
        ROIs_dir = fullfile(derivatives_dir, 'ROIs');
        wf_dir = fullfile(bids_dir, 'workflows');
        spDCM_dir = fullfile(derivatives_dir, 'spDCM');

        ptcs_tsv = fullfile(rawdata_dir,'participants.tsv');
        opts = detectImportOptions(ptcs_tsv, 'FileType','delimitedtext', 'Delimiter','\t');
        opts = setvartype(opts, 'SFB1280_code', 'char');
        subs_table = readtable(ptcs_tsv, opts);
        subs_table = subs_table((subs_table.rsfMRI==1)&(subs_table.T1w==1),:); % select subjects that have resting state data

        exc_subs_file = fullfile(base_dir, 'exclude_subs.csv');
        subs_table = exclude_subs(exc_subs_file, subs_table, AG, exp, 'spDCM');

        name_wf = sprintf('%s_%s',name_folder,name_model);
        wf_dir = fullfile(wf_dir, 'wf_spDCM', mri_space, name_wf);
        if ~exist(wf_dir,'dir'); mkdir(wf_dir); end

        run_group = 1;
        delete(gcp('nocreate')); c = parcluster; c.NumWorkers = 112; parpool(c);
        parfor sub_idx = 1:height(subs_table)

            ptc = subs_table(sub_idx,:).participant_id{:};
            sfbcode = subs_table(sub_idx,:).SFB1280_code{:};

            fprintf('\n\nProcessing subject: %s\n\n',ptc);

            p_wf_dir = fullfile(wf_dir, ptc);
            if ~exist(p_wf_dir,'dir'); mkdir(p_wf_dir); end

            argsnode = struct('p_wf_dir',p_wf_dir, 'rerun',0, 'name_node','');

            p_rawdata_dir = fullfile(rawdata_dir, ptc);

            % get TR and TE from rawdata func
            json_file = glob(fullfile(rawdata_dir, ptc,'func', '*task-rest*_bold.json'));
            txt = fileread(json_file{:});
            json_bold = jsondecode(txt);
            TR = json_bold.RepetitionTime;
            TE = json_bold.EchoTime;

            space_subdir = fullfile(spDCM_dir,ptc,mri_space,name_wf);
            if ~exist(space_subdir,'dir'), mkdir(space_subdir); end

            func_file = glob(fullfile(denoised_dir,ptc,'**',sprintf('*task-rest*space-%s*conftype-%s%s_bold.nii.gz',mri_space,confound_type,desc2)));
            mask_file = glob(fullfile(fmriprep_dir,ptc,'**',sprintf('*task-rest*space-%s_desc-brain_mask.nii.gz',mri_space)));

            info_ROIs = glob(fullfile(ROIs_dir, ptc, 'func', sprintf('*%s_dseg.json',name_folder)));
            fid = fopen(info_ROIs{:}, 'r');
            raw = fread(fid,inf);
            str = char(raw');
            fclose(fid);
            valid_ROIs = fieldnames(jsondecode(str));

            if network
                % Separate Left_* and Right_* ROIs
                left_ROIs = valid_ROIs(startsWith(valid_ROIs, 'Left_'));
                right_ROIs = valid_ROIs(startsWith(valid_ROIs, 'Right_'));

                % Get all pairs for Left_* and Right_* separately
                left_pairs = nchoosek(left_ROIs, 2);
                right_pairs = nchoosek(right_ROIs, 2);

                % Convert each row into a separate cell inside a cell array
                connections = [arrayfun(@(i) {left_pairs(i, :)}, 1:size(left_pairs, 1))'; ...
                    arrayfun(@(i) {right_pairs(i, :)}, 1:size(right_pairs, 1))'];
            end

            roi_name_hemi = split(valid_ROIs, '_');
            pats = cellfun(@(x,y) ['.*space-',mri_space,'_roi-',y,'_.*hemi-',x,'.*_mask.nii.gz'], roi_name_hemi(:,1),roi_name_hemi(:,2),'UniformOutput',false);
            pat = strjoin(pats,'|');

            allrois = glob(fullfile(ROIs_dir,ptc,'func',sprintf('*desc-%s*.nii*',name_folder))); %all rois
            srois = regexp(allrois, pat, 'match'); %selected rois
            vrois = [srois{:}]'; %valid rois

            files_gz = {func_file, mask_file, vrois};
            files_gunzip = run_node('run_gunzip', argsnode, files_gz);

            func = files_gunzip{1};
            mask = files_gunzip{2};
            rois = files_gunzip{3};
            % SPM specification
            glmdir = fullfile(space_subdir, ['GLM_',confound_type]);
            GLM_params = struct('glmdir',glmdir, 'units','scans', 'TR',TR, 'scans',func, 'hpf',Inf, 'mask',mask, 'mthresh',-Inf);
            spmmat = run_node('spdcm_GLM', argsnode, GLM_params);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % ROI extraction
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            VOI_dir = fullfile(space_subdir,['VOIs_',confound_type]);
            if ~exist(VOI_dir,'dir'), mkdir(VOI_dir); end
            cd(VOI_dir)

            DCM_VOI = run_node('spdcm_VOIs', argsnode, spmmat, rois, VOI_dir);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % SPECIFY & ESTIMATE DCM
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % preprend DCM_ to the filename
            DCM_filename = fullfile(space_subdir,sprintf('DCM_%s_space-%s_stat-DCM_conftype-%s_model.mat', ptc,mri_space,confound_type));
            [a_matrix,s] = create_a_matrix(DCM_VOI, connections);

            if s % if status is 1 (i.e., all connections are present), run DCM

                params_DCM = struct('TR',TR, 'TE',TE, 'DCM_filename',DCM_filename, 'a',a_matrix);
                params_DCM.options = struct('maxnodes', numel(rois));
                DCM = run_node('spdcm_DCM', argsnode, DCM_VOI, params_DCM);

                sel_conns = logical(reshape(a_matrix,1,[])');
                names_rois = [DCM.xY.name];
                combs = fliplr(gen_combs(names_rois,2,1));
                A_m = reshape(DCM.Ep.A,1,[])';
                ptc_t = repelem({ptc},sum(sel_conns))';
                AG_t = repelem({AG},sum(sel_conns))';
                exp_t = repelem({exp},sum(sel_conns))';
                Vp = full(diag(DCM.Cp));


                T = table(ptc_t, AG_t, exp_t, combs(sel_conns,1), combs(sel_conns,2), A_m(sel_conns), Vp(sel_conns), ...
                    'VariableNames',{'participant','AG','study','seed','target','spDCM','spDCM_Var'});

                DCM_df = fullfile(space_subdir,sprintf('%s_space-%s_stat-DCM_conftype-%s_df.tsv', ptc,mri_space,confound_type));

                writetable(T, DCM_df, 'FileType','text', 'Delimiter','\t');
            end
        end

        if run_group
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % % First provide estimated DCM as GCM (Group DCM) cell array. Individual DCMs can be
            % % estimated by using spm_dcm_fit.m

            GCM = glob(fullfile(spDCM_dir, sprintf('sub-**/%s/*_space-%s_stat-DCM_conftype-%s_model.mat',name_wf,mri_space,confound_type)));

            wf_group_dir = fullfile(wf_dir, 'group', mri_space, name_wf);
            if ~exist(wf_group_dir,'dir'), mkdir(wf_group_dir); end

            argsnode = struct('p_wf_dir',wf_group_dir, 'rerun',0, 'name_node','');

            group_results_dir = fullfile(spDCM_dir, 'group', mri_space, name_wf);
            if ~exist(group_results_dir,'dir'), mkdir(group_results_dir); end

            modelfile = fullfile(group_results_dir,sprintf('DCM_space-%s_conftype-%s_model.mat',mri_space,confound_type));

            gDCM = run_node('spdcm_PEB_BMC', argsnode, GCM, modelfile);


            GCM = glob(fullfile(spDCM_dir, sprintf('sub-**/%s/*_space-%s_stat-DCM_conftype-%s_df.tsv',name_wf,mri_space,confound_type)));
            t_group = table();
            for t = 1:length(GCM)
                t_temp = readtable(GCM{t}, 'FileType','text','Delimiter','\t');
                t_group = [t_group;t_temp];
            end

            t_group(:, {'seeds','targs'}) = cellfun(@(x) strsplit(x,'_'), [t_group.seed,t_group.target], 'UniformOutput', false);
            t_group = splitvars(t_group, {'seeds','targs'}, 'NewVariableNames',{{'hemi_seed','roi_seed'},{'hemi_target','roi_target'}});

            group_df = fullfile(group_results_dir,sprintf('space-%s_stat-DCM_conftype-%s_df.tsv',mri_space,confound_type));
            writetable(t_group, group_df, 'FileType','text', 'Delimiter','\t');
        end
    end
end

function outargs = spdcm_unzip(func_file, mask_file)

func = glob(func_file);
if isempty(func)
    warning('%s does not exist!',func_file);
    return;
end
funcs = gunzip(func{:});
mask = glob(mask_file);
if isempty(func)
    warning('%s does not exist!',mask);
    return;
end
mask = gunzip(mask{:});

outargs.func = funcs;
outargs.mask = mask;


function outargs = spdcm_VOIs(spmmat, rois, VOI_dir)

if isempty(rois)
    error('No valid ROIs!')
end

nb_rois = length(rois);
for idx_roi = 1:nb_rois
        
        [~,filename,~] = fileparts(rois{idx_roi});

        newname = join([extractBetween(filename,'hemi-','_'), extractBetween(filename,'roi-','_')],'_');

        matlabbatch_VOI{1}.spm.util.voi.spmmat = {spmmat};
        matlabbatch_VOI{1}.spm.util.voi.adjust = 0;
        matlabbatch_VOI{1}.spm.util.voi.session = 1;
        matlabbatch_VOI{1}.spm.util.voi.name = fullfile(VOI_dir,filename);
        matlabbatch_VOI{1}.spm.util.voi.roi{1}.label.image = rois(idx_roi);
        matlabbatch_VOI{1}.spm.util.voi.roi{1}.label.list = 1;
        matlabbatch_VOI{1}.spm.util.voi.expression = 'i1';
        
        spm_jobman('run',matlabbatch_VOI);
        
        % Add VOIs to DCM structure
        name_VOI = sprintf('VOI_%s_1.mat', filename);
        VOI = load(fullfile(VOI_dir, name_VOI));
        DCM_VOI.xY(idx_roi) = VOI.xY;
        DCM_VOI.xY(idx_roi).name = newname;
        
end

outargs.DCM_VOI = DCM_VOI;

function outargs = spdcm_GLM(GLM_params)

    contents_dir = dir(GLM_params.glmdir);
    if ~isempty(contents_dir)
        rmdir(GLM_params.glmdir,'s');
        mkdir(GLM_params.glmdir)
    end
    matlabbatch{1}.spm.stats.fmri_spec.dir = {GLM_params.glmdir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = GLM_params.units;
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = GLM_params.TR;
    if isfield(GLM_params, 'confounds_file')
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {GLM_params.confounds_file};
    end
    matlabbatch{1}.spm.stats.fmri_spec.sess.scans = {GLM_params.scans};
    matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = GLM_params.hpf;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {GLM_params.mask};
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = GLM_params.mthresh;
    
    % SPM estimation
    spmmat = fullfile(GLM_params.glmdir,'SPM.mat');
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {spmmat};
    
    spm_jobman('run',matlabbatch);
    
    outargs.spmmat = spmmat;
	
function [a,s] = create_a_matrix(DCM_VOI, connections, self_connections, reverse_rois)


arguments
    DCM_VOI (1,1) struct
    connections (1,:) cell
    self_connections (1,1) logical = true
    
    % reverse_rois simply selects the connections in the opposite
    % direction, e.g., if HIP->AMY is a connection of interest, if
    % reverse_rois is true, then the connection AMY->HIP will also be
    % included in the "a" matrix
    reverse_rois (1,1) logical = true
end

n = length(DCM_VOI.xY);      % number of regions

if self_connections
    a = eye(n,n);
else
    a = zeros(n,n);
end

s = 1;
rois = [DCM_VOI.xY.name];
for conn = 1:numel(connections)
        idx_c = find(strcmp(rois,connections{conn}{1}));
        idx_r = find(strcmp(rois,connections{conn}{2}));
        
        % if some of the connections don't exist s should be 0 because
        % these subs will need to be excluded
        if isempty(a(idx_r,idx_c)); s = 0; end 
        
        a(idx_r,idx_c) = 1;
        if reverse_rois
            a(idx_c,idx_r) = 1;
        end
end
	
function outargs = spdcm_DCM(DCM, params_DCM)

% Metadata
v = length(DCM.xY(1).u); % number of time points
n = length(DCM.xY);      % number of regions

DCM.v = v;
DCM.n = n;

% Timeseries
DCM.Y.dt  = params_DCM.TR;
DCM.Y.X0  = DCM.xY(1).X0;
DCM.Y.Q   = spm_Ce(ones(1,n)*v);
for i = 1:DCM.n
    DCM.Y.y(:,i)  = DCM.xY(i).u;
    DCM.Y.name{i} = DCM.xY(i).name;
end

% Task inputs
DCM.U.u    = zeros(v,1);
DCM.U.name = {'null'};

% Connectivity
if ~isfield(params_DCM,'a')
    params_DCM.a = ones(n,n);
end
DCM.a = params_DCM.a;
DCM.b = zeros(n,n,0);
DCM.c = zeros(n,0);
DCM.d = zeros(n,n,0);

% Timing
DCM.TE = params_DCM.TE;
DCM.delays = repmat(params_DCM.TR,DCM.n,1);

% Options
DCM.options.nonlinear = 0;
DCM.options.two_state = 0;
DCM.options.stochastic = 0;
DCM.options.analysis = 'CSD';
DCM.options.induced = 1;

% check if other options have been specified
for fn = fieldnames(params_DCM.options)'
   DCM.options.(fn{1}) = params_DCM.options.(fn{1});
end

[base,~,~] = fileparts(params_DCM.DCM_filename);

if ~exist(base,'dir'), mkdir(base); end
DCM.dir = params_DCM.DCM_filename;
save(DCM.dir, 'DCM');

DCM = spm_dcm_fmri_csd(DCM.dir);

outargs.DCM = DCM;


function outargs = spdcm_PEB_BMC(GCM, modelfile)

M = struct();
M.alpha = 1;
M.beta  = 16;
M.hE    = 0;
M.hC    = 1/16;
M.Q     = 'single';

N = length(GCM);

% Specify design matrix for N subjects. It should start with a constant column
M.X = ones(N,1);

% Choose field
field = {'A'};

% Estimate model
PEB = spm_dcm_peb(GCM,M,field);
[BMA, RCM] = spm_dcm_peb_bmc(PEB);

save(modelfile, 'GCM','PEB','BMA','RCM');

spm_dcm_peb_review(BMA,{GCM});

outargs.modelfile = modelfile;
