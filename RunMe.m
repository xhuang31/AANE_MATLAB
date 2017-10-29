% Code for Accelerated Attributed Network Embedding
% 
%   Copyright 2017, Xiao Huang and Jundong Li.
%   $Revision: 1.0.0 $  $Date: 2017/10/18 00:00:00 $

%% Load data
load('BlogCatalog.mat')
lambda = 10^-0.6; % the regularization parameter
rho = 5; % the penalty parameter
% load('Flickr.mat')
% lambda = 0.0425; % the regularization parameter
% rho = 4; % the penalty parameter

%% Experimental Settings
d = 100; % the dimension of the embedding representation
G = Network; % the weighted adjacency matrix
A = Attributes; % the attribute information matrix with row denotes nodes
clear Attributes & Network
Indices = crossvalind('Kfold',length(G),25); % 5-fold cross-validation indices
Group1 = find(Indices <= 20); % 2 for 10%, 5 for 25%, 10 for 50%, 20 for 100% of training group
Group2 = find(Indices >= 21); % test group, test each fold in turns
n1 = length(Group1); % num of nodes in training group
n2 = length(Group2);  % num of nodes in test group
CombG = sparse(G([Group1;Group2],[Group1;Group2]));
CombA = sparse(A([Group1;Group2],:));


%% Accelerated Attributed Network Embedding
disp('Accelerated Attributed Network Embedding (AANE), 5-fold with 100% of training is used:')
tic
H = AANE_fun(sparse(CombG),sparse(CombA),d,lambda,rho);
toc
[F1macro1,F1micro1] = Performance(H(1:n1,:),H(n1+1:n1+n2,:),Label(Group1,:),Label(Group2,:)) %


%% AANE for a Pure Network
disp('AANE for a pure network:')
tic
H = AANE_fun(sparse(CombG),sparse(CombG),d,lambda,rho);
toc
[F1macro2,F1micro2] = Performance(H(1:n1,:),H(n1+1:n1+n2,:),Label(Group1,:),Label(Group2,:)) %
