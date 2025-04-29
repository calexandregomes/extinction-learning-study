library(ggplot2)
library(ggcorrplot)
library(dplyr)
library(stringr)
library(tibble)
library(tidyr)
library(lmerTest)

base_dir = "G://F02"
fig_dir =  file.path("G://figures")
learning_dir =  file.path(base_dir, "group", "learning")

AGs = list(
  A02 = list(
    "Extinction_EEG_fMRI" = list(
      acquisition = c(stat='DCM', model='DelayAE0FixConst', col='Fixed.response...2..response.amplitude'),
      extinction = c(stat='DCM', model='DelayAE0FixConst', col='Fixed.response...2..response.amplitude'))),
  A03 = list(
    "3T" = list(
      acquisition = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
      extinction = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
      renewal = c(stat='DCM', model='DelayRen2p5Const', col='Flexible.response...1..amplitude'),
  "3T_EEG" = list(
      acquisition = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
      extinction = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
      renewal = c(stat='DCM', model='DelayRen2p5Const', col='Flexible.response...1..amplitude'),
  A05 = list(
    "study3" = list(
      acquisition = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
      extinction = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
  A09 = list(
    "Extinction_Generalization_I" = list(
      acquisition = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
      extinction = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
    "Extinction_Generalization_II" = list(
      acquisition = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
      extinction = c(stat='DCM', model='DelayAE25Const', col='Flexible.response...1..amplitude'),
  A12 = list(
    "study_1" = list(
      acquisition = c(stat='SF', model='breakmissLongAcqsum', col='amplitude'),
      extinction = c(stat='DCM', model='DelayE25Const', col='Flexible.response...1..amplitude')),
    "study_2" = list(
      acquisition = c(stat='SF', model='breakmissLongAcqsum', col='amplitude'),
      extinction = c(stat='DCM', model='DelayE25Const', col='Flexible.response...1..amplitude'))))

df_stats = data.frame()
for (AG in names(AGs)) {
  for (study in names(AGs[[AG]])) {
    for (task in names(AGs[[AG]][[study]])) {
      
      v = AGs[[AG]][[study]][[task]]
      
      pspm_derivatives_dir = file.path(base_dir, AG, study, "derivatives", "pspm", "group")
      p = file.path(pspm_derivatives_dir, sprintf('stat-%s_type-betas_desc-%s_df.tsv',v[["stat"]],v[["model"]]))
      df_stats_temp = read.csv(p, sep='\t')
      
      scols = c("participant","AG","study","cond","task","session","trial_task","trial_cond","amplitude")
      if (AG=="A02") {scols = c(scols,"CS_type")}
      df_stats_temp2 = df_stats_temp %>% filter(task==!!task) %>% rename(c("amplitude"=!!v[["col"]])) %>% dplyr::select(any_of(scols))
      
      df_stats=bind_rows(df_stats,df_stats_temp2)  
    }
  }
}

study1s = c("Extinction_EEG_fMRI","3T","study3","Extinction_Generalization_I","study_1")
df_stats2 = df_stats %>% mutate(study2=if_else(study %in% study1s, "study1", "study2"))

s = c('_G','_G2','_G3','_G4','_G0','_Auditive','mid_ACQ','end_ACQ','mid_EXT','end_EXT')
idxs1 = Reduce('|', lapply(s, function(y) endsWith(as.character(df_stats2$cond), y)))
idxs2 = as.character(df_stats2$cond) %in% c('context','Context','noUS','noUS+','US')
df_stats2 = df_stats2[!(idxs1|idxs2),]

df_stats2$cond2 = factor(ifelse(startsWith(as.character(df_stats2$cond),'CS+'), 'CS+', 'CS-'))

df_stats2 = df_stats2 %>% group_by(participant,AG,study,cond2,task) %>% mutate(trial_cond2=sequence(n()))

if ("A02" %in% df_stats2$AG) {
  df_stats2 = df_stats2 %>% filter( !((task=="extinction")&(CS_type %in% c(2,3))))
}

diff_cond = df_stats_AC %>% group_by(participant,AG,study,cond) %>% summarise(amplitude=mean(amplitude)

df_stats_AC = subset(df_stats2, (task=="acquisition"))
df_stats_EX = subset(df_stats2, (task=="extinction"))
df_stats_REN = subset(df_stats2, (task=="renewal"))

df_stats_short = df_stats2 %>% ungroup %>% group_by(participant,AG,study,task) %>% 
  mutate(trial_cond2 = ave(cond2==cond2, cond2, FUN=\(x) cut(seq_along(x), 8))) %>% 
  group_by(participant,AG,study,cond2,trial_cond2,task) %>% summarise(amplitude = mean(amplitude))
df_stats_EX_short = df_stats_EX %>% ungroup() %>% group_by(participant,AG,study) %>% 
  mutate(trial_cond2 = ave(cond2==cond2, cond2, FUN=\(x) cut(seq_along(x), 8))) %>%
  group_by(participant,AG,study,cond2,trial_cond2) %>% summarise(amplitude = mean(amplitude)) 
df_stats_RE_short = df_stats_REN %>% ungroup() %>% group_by(participant,AG,study) %>% 
  mutate(trial_cond2 = ave(cond2==cond2, cond2, FUN=\(x) cut(seq_along(x), 8))) %>%
  group_by(participant,AG,study,cond2,trial_cond2) %>% summarise(amplitude = mean(amplitude)) 

df_stats_short = subset(df_stats_AC, AG!="A02")
a = dplyr::select(subset(df_stats, (AG=="A09")&(task=="acquisition")), c("participant","study","cond","trial_cond"))
df_stats_short = df_stats_short %>% left_join(a, by=c("participant","study","trial_cond"))
df_stats_short = df_stats_short %>%  mutate(cond=case_when( cond=="CS+noUS" ~ "CS+noUS",
                                         cond=="CS+noUS_Visceral" ~ "CS+noUS",
                                         cond=="CS+US" ~ "CS+US",
                                         cond=="CS+US_Visceral" ~ "CS+US",
                                         cond=="CS+US_N"~ "CS+US",
                                         cond=="CS+noUS_N" ~ "CS+noUS",
                                         .default = "NA"))
 
m = lmer(amplitude ~ cond2*trial_cond2 + (1|participant), data=df_stats_RE_short)
summary(m)
em = emtrends(m, ~cond2, var="trial_cond2")
pairs(em)

ptcs = unique(df_stats2[,c("participant","AG","study","task")])
######################
#### RUN BF MODEL ####
######################
df_bf = data.frame(participant=factor(), AG=factor(), study=factor(), cond=factor(), task=factor(), Intercept=integer(), 
                Slope=integer(), AIC=integer())
for (idx in 1:nrow(ptcs)) {

  df_sub = df_stats2 %>% filter(participant %in% ptcs[[idx,'participant']] & AG %in% ptcs[[idx,'AG']] & 
                                 study %in% ptcs[[idx,'study']] & task %in% ptcs[[idx,'task']])
  df_sub = df_sub[,colSums(is.na(df_sub))<nrow(df_sub)]
  df_sub = df_sub[complete.cases(df_sub),]
  
  if (task %in% c("recall","renewal")) {
    r = df_sub %>% group_by(cond2) %>% summarise(mean=mean(amplitude))
    vCSp = r[r$cond2=="CS+",]$mean
    vCSm = r[r$cond2=="CS-",]$mean
    
  }
  else {
    m = lm(amplitude ~ poly(trial_cond2,2)*cond2, data=df_sub)
  
    df_sub$pred = predict(m)
    dCSp = diff((df_sub[df_sub$cond2=='CS+','pred'])$pred)
    dCSm = diff((df_sub[df_sub$cond2=='CS-','pred'])$pred)

  sizeCSp = c(floor(length(dCSp)/2), c(length(dCSp) - c(floor(length(dCSp)/2))) )
  sizeCSm = c(floor(length(dCSm)/2), c(length(dCSm) - c(floor(length(dCSm)/2))) )
  
  wCSp = c(rep(0.7/sizeCSp[1], times=sizeCSp[1]), rep(0.3/sizeCSp[2], times=sizeCSp[2]))
  wCSm = c(rep(0.7/sizeCSm[1], times=sizeCSm[1]), rep(0.3/sizeCSm[2], times=sizeCSm[2]))
  
  vCSp = sum(dCSp)
  vCSm = sum(dCSm)
  }
  
  df_bf = df_bf %>% add_row(participant=ptcs[[idx,'participant']], AG=ptcs[[idx,'AG']], study=ptcs[[idx,'study']], cond="CS-", 
                      task=ptcs[[idx,'task']], Slope=vCSm) %>% 
    add_row(participant=ptcs[[idx,'participant']], AG=ptcs[[idx,'AG']], study=ptcs[[idx,'study']], cond="CS+", 
            task=ptcs[[idx,'task']], Slope=vCSp)
  
}

# swap sign for extinction, renewal and recall, so that greater values indicate greater learning
idxs = df_bf$task %in% c("extinction")
df_bf[idxs,'Slope'] = -df_bf[idxs,'Slope']
df2 = df_bf %>% pivot_wider(names_from=cond, values_from=c(Slope))

df2$learning = df2[['CS+']] - df2[['CS-']]
learning = df2[c("participant","AG","study","task","learning")]

#################################################################################################
############################# BEHAVIOURAL LEARNING ##############################################

base_dir = "G://F02"
fig_dir =  file.path("G://figures")

learning_dir =  file.path("G://F02//group//learning")

pred_t = function(p,b0,b1,df=NULL,col="prob",rows=NULL) {
  t = (-b0 + log(p/(1-p)) )/b1
  return(t)
}

pred_ev = function(col,rows=NULL) {
  if (is.null(rows)) {
    rows = 1:length(col)
  }
  ev = sum( col[rows] )
  return(ev)
}

dat = read.csv('G:\\F02\\A08\\desc-behaviour_df.tsv', sep='\t')
df_diffCSs = read.csv(file.path(learning_dir, "desc-A08diffCS_df.tsv"), sep='\t')

# ACQUISITION
acq = subset(dat, task=="Acquisition")
m = glmer(accuracy~trial_cond + session + (trial_cond|participant), data=acq, family='binomial')
df = coef(m)$participant
names(df) = c("intercept","trial","session")
df = tibble::rownames_to_column(df, "participant") %>% mutate(col=if_else(trial<0,"Negative","Positive"))

mean_acc = group_by(acq,participant) %>% summarise(ACQ_acc=mean(accuracy))
df = merge(df,mean_acc[,c("participant","ACQ_acc")], by="participant")

dat2 = unique(acq[,c("participant","trial_cond","session")])
dat2$prob = predict(m, newdata=dat2, type="response")
# in case there is more than one session, average probabilities across sessions
dat2 = dat2 %>% group_by(participant,trial_cond) %>% summarise(prob=mean(prob))

evs_acq= dat2 %>% group_by(participant) %>% summarise(evs_acq=pred_ev(prob,rows=1:8))

df_acq = merge(df,evs_acq[,c("participant","evs_acq")], by="participant")
df_acq = merge(df_acq,unique(acq[,c("participant","study")]), by="participant")
df_acq$AG = 'A08'
df_acq$task = 'acquisition'

df1 = df_acq[c('participant','AG','study','task','evs_acq')]
df_ACQ = rename(df1, c('learning'='evs_acq'))


# EXTINCTION
ext = subset(dat, (task=="Extinction") & (trial_type=="EX"))
m = glmer(accuracy~trial_cond + session + (trial_cond|participant), data=ext, family='binomial')
df = coef(m)$participant
names(df) = c("intercept","trial","session")
df = tibble::rownames_to_column(df, "participant") %>% mutate(col=if_else(trial<0,"Negative","Positive"))

mean_acc = group_by(ext,participant) %>% summarise(EXT_acc=mean(accuracy))
df = merge(df,mean_acc[,c("participant","EXT_acc")], by="participant")

dat2 = unique(ext[,c("participant","trial_cond","session")])
dat2$prob = predict(m, newdata=dat2, type="response")
dat2 = dat2 %>% group_by(participant,trial_cond) %>% summarise(prob=mean(prob))

evs_ext=group_by(dat2, participant) %>% summarise(evs_ext=pred_ev(prob,rows=1:8))

df_ext = merge(df,evs_ext[,c("participant","evs_ext")], by="participant")
df_ext = merge(df_ext,unique(acq[,c("participant","study")]), by="participant")
df_ext$AG = 'A08'
df_ext$task = 'extinction'

df2 = df_ext[c('participant','AG','study','task','evs_ext')]
df_EXT = rename(df2, c('learning'='evs_ext'))

# RENEWAL
ren = subset(dat, (task=="Renewal") & (trial_type=="EX")) #& (stimulus_type %in% c("E","F")))

ren = ren %>% mutate(renewal = case_when(
  response_key==5 ~ 1, #same response as acquisition
  response_key==6 ~ 0,
  .default = NA))

m = glmer(renewal ~ trial_cond + session + (trial_cond|participant), data=subset(ren, stimulus_type %in% c("E","F")), family='binomial')
summary(m)
df = coef(m)$participant
names(df) = c('intercept','trial','session')
df = tibble::rownames_to_column(df, "participant") %>% mutate(col=if_else(trial<0,"Negative","Positive"))

dat2 = unique(ren[,c("participant","trial_cond","session")])
dat2$prob = predict(m, newdata=dat2, type="response")
dat2 = dat2 %>% group_by(participant,trial_cond) %>% summarise(prob=mean(prob))

evs_ren=group_by(dat2, participant) %>% summarise(evs_ren=pred_ev(prob,rows=1:5))

df_ren = merge(df,evs_ren[,c("participant","evs_ren")], by="participant")
df_ren = merge(df_ren,unique(acq[,c("participant","study")]), by="participant")
df_ren$AG = 'A08'
df_ren$task = 'renewal'

df2 = df_ren[c('participant','AG','study','task','evs_ren')]
df_REN = rename(df2, c('learning'='evs_ren'))

##################################################################

df_all = rbind(df_ACQ,df_EXT,df_REN)
df_all$task = recode(df_all$task, acquisition="Acquisition",extinction="Extinction",renewal="Renewal")
