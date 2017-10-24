% Surrogate Parameters
% 0 is adaptive cluster
param.surr_num_clusters = 1;
% Type of approximation methods
param.surr_type = {'dace' 'rbf' 'rsm2' 'rsm1' 'mlp'};
% Takes the last 1000 from the archive
param.surr_max_traincount = 1000;
% Training and validation 
param.surr_train_ratio = 0.8;
% Local SUBEA(NSGA-II) Parameters
param.subea_pop_size=100;
param.subea_generations=20;
% Problem specific inputs
% param.config = {'zdt1','zdt2','zdt3','osy','srn','tnk','alkyl_a','alkyl_b'};
param.config = {'zdt1','osy','srn','test_tan','welded_beam','Car_Crash','bulk_carrier_design'};
param.pres_func_evalall = [1200;2000;1100;600;900;1000;2000];
param.popsize_prob = [22;22;12;30;20;30;28]; % Population size for each problem
param.gen_prob = 5000000*ones(length(param.config),1); % Set the number of generation a high value
param.crossover_pr_prob = 0.9*ones(length(param.config),1);
param.mutation_pr_prob = 0.1*ones(length(param.config),1);
param.crossover_sbx_prob = 10*ones(length(param.config),1);
param.mutation_poly_prob = 20*ones(length(param.config),1);
param.seed_prob = 199*ones(length(param.config),1);
param.multirun = 10*ones(length(param.config),1); % For number of runs taking different seeds