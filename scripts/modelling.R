library(dplyr)
library(tidyr)
library(tibble)
library(glmnet)
library(caret)
library(emmeans)
library(car)
require(doMC)

cl <- parallel::makeCluster(24)
doParallel::registerDoParallel(cl)

norm = 1

tasK = "AC"

root_dir = "G:"
base_dir = file.path(root_dir, "F02")
group_dir =  file.path(base_dir, "group")
fig_dir =  file.path(root_dir,"figures")

% load helper functions: normData, scale2, composite_score, fit_lasso, etc.
source(file.path(root_dir,"scripts","R","misc_funs.R"))

ptc_files = Sys.glob(file.path(base_dir,"A*","*","rawdata","participants.tsv"))

dat_learn_EDA = read.csv(file.path(group_dir, "learning", "stat-bf_desc-EDA_df.tsv"), sep='\t')

dat_learn_beh = read.csv(file.path(group_dir, "learning", "stat-logit_desc-beh_df.tsv"), sep='\t')
dat_learn = bind_rows(dat_learn_EDA, dat_learn_beh) %>% filter(!is.na(learning))

covariates = c("age","sex")

demographics = get_demogs(ptc_files, c("participant_id","AG","study",covariates))

comb_AG = gen_combs(unique(dat_learn$AG))
comb_AG = list(c("A08"), # PL
			         c("A03"), # FLr
               c("A03","A05"), # FLc
               c("A03","A05","A09"),# FLs
			         c("A02","A03","A05","A09"), # FLel
               c("A03","A05","A09","A12"), # FLst
               c("A02","A03","A05","A09","A12"), # FLa
´              c("A02","A03","A05","A08","A09","A12")) # All


dupls = read.csv(file.path(base_dir, "desc-duplicates_df.tsv"), sep='\t')
excl_subs = dupls[dupls$keep==0,c("participant","AG","study")]
dat_learn = anti_join(dat_learn, excl_subs, by=c("participant","AG","study"))

dat_learn_corrs = dat_learn %>% group_by(AG,study,task) %>% group_modify(~ normData(.x))

df_learndemo = demographics %>% rename(participant=participant_id) %>%
right_join(dat_learn, by=c("participant","AG","study"), multiple='all')

dat_excl_conn = read.csv(file.path(base_dir, "desc-ExcludeSubsConn_table.tsv"), sep='\t')

########################################################################################
############################### CONNECTIVITY ###########################################
########################################################################################
dat_FC = read.csv(file.path(group_dir, "FC", "desc-FC_df.tsv"), sep='\t')
dat_FC = subset(dat_FC, !(participant %in% dat_excl_conn[dat_excl_conn$pipeline=="FC",]$participant_id))

dat_DTI = read.csv(file.path(group_dir, "SC", "desc-SC_df.tsv"), sep='\t')
dat_DTI = subset(dat_DTI, !(participant %in% dat_excl_conn[dat_excl_conn$pipeline=="DTI",]$participant_id))

dat_spDCM = read.csv(file.path(group_dir,"EC","desc-EC_df.tsv"), sep='\t')
dat_spDCM = subset(dat_spDCM, !(participant %in% dat_excl_conn[dat_excl_conn$pipeline=="spDCM",]$participant_id))

df_comb = data.frame(); df_coefs = data.frame(); df_pred = data.frame();

l = list(FC=dat_FC, SC=dat_DTI, EC=dat_spDCM)
for (nAG in 1:length(comb_AG)) {
  AGs = comb_AG[[nAG]]

  for (n in 1:length(l)) {

    if (names(l[n]) == 'FC') {
      metric = c('corrLW','xcorr','EuclideanDist','ManhattanDist','WassersteinDist','dtw','MI','mscohe','wavcohe')
      inv = c('EuclideanDist','ManhattanDist','WassersteinDist','dtw')
      absl = c('corrLW')
      dat_conn = dat_FC
      pair='pair_und'
    } else if (names(l[n]) == 'SC') {
      metric = 'streamlines'
      dat_conn = dat_DTI
      pair='pair_und'
    } else {
      metric = 'spDCM'
      dat_conn = dat_spDCM
      pair='pair_dir'
    }

    dat_conn = subset(dat_conn, hemisphere!="bilateral")
    dat_conn[dat_conn$participant=="sub-VA05IJ10","study"] = "Extinction_Generalization_I"

    df_join = right_join(df_learndemo, dat_conn, by=c("participant","AG","study"), multiple='all')

    ####### CHOOSE WHICH AGs TO INCLUDE IN THE ANALYSIS HERE #######
    df_join = df_join %>% subset(AG %in% AGs)
    if (dim(df_join)[1]==0) {next}
    ################################################################

    df_sel = df_join %>% group_by(participant, AG, study) %>%
      filter(if_all(!!metric, ~ all(!is.na(.x))))

    # df_sel will contain the final sample - that is, excluding subs with any NAs
    # Calculate the composite score
    if (names(l[n]) == 'FC') {
      df_sel = composite_score(df_sel, cols=metric, inv=inv, absl=absl, keep_ori="corrLW")
      metric = "composite"
    }
    
    selvars = unique(c("participant","AG","study","task",pair,metric,"learning",covariates))

	# Normalisation
    df = df_sel[selvars] %>% group_by(AG,study,task,across(all_of(pair))) %>% group_modify(~ normData(.x)) %>%
        pivot_wider(names_from=all_of(pair), values_from=all_of(metric), values_fn=mean) %>% ungroup()
    
    if (names(l[n]) == 'EC') {
      mVp = df_sel %>% group_by(participant, AG, study) %>% mutate(mVp=mean(spDCM_Var)) %>%
        select(participant, AG, study, mVp) %>% distinct()
      mVp$mVp = 1-range01(mVp$mVp)
      df = left_join(df, select(mVp, mVp))
    }
    
    df$const <- factor(rep(1, each=length(df$participant)))

    df_acq = subset(df, task == "acquisition")
    df_ext = subset(df, task == "extinction")
    df_ren = subset(df, task == "renewal")

    if (tasK=="AC") {
      mdf = df_acq
    } else if (tasK=="EX") {
      mdf = df_ext
    } else {
	  mdf = df_ren

    pairs = colnames(mdf %>% dplyr::select(starts_with(
      c('AMY','CEB','HIP','ACC','PFC','lAMY','lCEB','lHIP','lACC','lPFC','rAMY','rCEB','rHIP','rACC','rPFC'))))

    #################################################################
    #################### LASSO ######################################
    #################################################################
    y <- mdf$learning
    xx <- mdf %>% ungroup() %>% dplyr::select(all_of(c(pairs,covariates)))
    x = data.matrix(makeX(xx, na.impute = TRUE))
    myalpha = 1

    if (!is.null(covariates)) {
      force.vars = as.integer(!Reduce('|', lapply(covariates, function(y) startsWith(as.character(colnames(x)), y))))
    } else {
      force.vars = rep(1, ncol(xx))
    }

    tymea = "mse"
    lambda_max <- max(abs(colSums(x*y,na.rm=T)))/nrow(x)
    epsilon <- .0001
    K <- 1000
    lambdapath <- round(exp(seq(log(lambda_max), log(lambda_max*epsilon), length.out = K)), digits = 10)
    lbpath = lambdapath

    ls = foreach(i = 1:100, .combine='rbind', .packages="glmnet") %dopar% {
      fit <- cv.glmnet(x, y, alpha=myalpha, nfolds=10, standardize=T, penalty.factor=force.vars,
                       type.measure=tymea, parallel=T, lambda=lbpath)
      errors = data.frame(fit$lambda,fit$cvm)
    }

    ls <- aggregate(ls[, 2], list(ls$fit.lambda), mean)

    bestindex = which(ls[2]==min(ls[2]))[1]
    lbd = ls[bestindex,1]

    if (names(l[n]) == 'EC') {
      ws = 1-mdf$mVp
    } else {ws = NULL}
    
    fit.lasso = fit_lasso(x=x,y=y,lambda=lbd, alpha=myalpha, penalty.factor=force.vars, type.measure=tymea,
                       weights=ws)
    results = coef(fit.lasso)

    yhat = predict(fit.lasso, newx=x)

    temp = data.frame(modality=names(l[n]), yhat=yhat, y=y)
    df_pred = rbind(df_pred, temp)

    preds = rownames(results)
    exclpreds = preds[grepl(paste(c('(Intercept)',covariates), collapse="|"),preds)]

    df_res = as.data.frame(as.matrix(results)) %>%
      filter_all(any_vars(.!=0)) %>% rownames_to_column(var="connection") %>%
      filter(!(connection %in% exclpreds))
    if (dim(df_res)[1]==0) {df_res[1,]=NA}
    df_res['modality'] = names(l[n])
    df_res['groups'] = paste(AGs,collapse='_')
    df_res['N'] = nrow(mdf)
                                                  
    df_comb = rbind(df_comb,df_res)

  } # end of for loop for comb_AG
  
  ###########################################################
  # calculate generalisibility using mean squared errors
  ###########################################################
  
  tp = foreach(niter = 1:100, .combine='rbind', .packages=c("dplyr","doMC","tibble","glmnet")) %dopar% {
    mdf2 = mdf[sample(nrow(mdf), 80), 
    #################################################################
    #################### LASSO ######################################
    #################################################################
    y <- mdf2$learning
    xx <- mdf2 %>% ungroup() %>% dplyr::select(all_of(c(pairs,covariates)))
    x = data.matrix(makeX(xx, na.impute = TRUE))
    myalpha = 1
    
    if (!is.null(covariates)) {
      force.vars = as.integer(!Reduce('|', lapply(covariates, function(y) startsWith(as.character(colnames(x)), y))))
    } else {
      force.vars = rep(1, ncol(xx))
    }
    
    #####################
    ##### Run lasso
    #####################
    tymea = "mse"
    lambda_max <- max(abs(colSums(x*y,na.rm=T)))/nrow(x)
    epsilon <- .0001
    K <- 1000
    lambdapath <- round(exp(seq(log(lambda_max), log(lambda_max*epsilon), length.out = K)), digits = 10)
    lbpath = lambdapath
    
    ls = foreach(i = 1:100, .combine='rbind', .packages="glmnet") %dopar% {
      fit <- cv.glmnet(x, y, alpha=myalpha, nfolds=10, standardize=T, penalty.factor=force.vars,
                       type.measure=tymea, parallel=T, lambda=lbpath)
      errors = data.frame(fit$lambda,fit$cvm)
    }
    
    ls <- aggregate(ls[, 2], list(ls$fit.lambda), mean)
    
    bestindex = which(lambdas[2]==min(lambdas[2]))[1]
    lbd = ls[bestindex,1]
    
    if (names(l[n]) == 'EC') {
      ws = 1-mdf2$mVp
    } else {ws = NULL}
    
    fit.lasso = fit_lasso(x=x,y=y,lambda=lbd, alpha=myalpha, penalty.factor=force.vars, type.measure=tymea,
                       weights=ws)
    results = coef(fit.lasso)
    
    df_res = as.data.frame(as.matrix(results)) %>% rownames_to_column(var="connection")
    
    preds = df_res$connection
    
    df_tempc = data.frame(G = niter, AMY=length(grep("AMY", preds)), ACC=length(grep("ACC", preds)), HIP=length(grep("HIP", preds)), PFC=length(grep("PFC", preds)),
                          CEB=length(grep("CEB", preds)), mod=names(l[n]))
    
  } 
  df_counts = rbind(df_counts,tp)
}

}

m = df_counts %>% pivot_longer(cols = -c(G,mod)) %>% group_by(G,name)
  
model <- glmer(value ~ name + (1|G), family=poisson, data=subset(m, mod==mg))
summary(model)
em = emmeans(model, pairwise ~ name, adjust="fdr")
