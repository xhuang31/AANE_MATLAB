## Accelerated Attributed Network Embedding
- Accelerated Attributed Network Embedding, SDM 2017
- A General Embedding Framework for Heterogeneous Information Learning in Large-Scale Networks, TKDD, 2018


## Code in MATLAB
```
H = AANE_fun(Net,Attri,d);  
H = AANE_fun(Net,Attri,d,lambda,rho);  
H = AANE_fun(Net,Attri,d,lambda,rho,'Att');  
H = AANE_fun(Net,Attri,d,lambda,rho,'Att',splitnum);  
```

- H is the joint embedding representation of Net and Attri;
- Net is the weighted adjacency matrix;
- Attri is the node attribute information matrix with row denotes nodes.


## Reference in BibTeX: 
@conference{Huang-etal17Accelerated,  
Author = {Xiao Huang and Jundong Li and Xia Hu},  
Booktitle = {SIAM International Conference on Data Mining},  
Pages = {633--641},  
Title = {Accelerated Attributed Network Embedding},  
Year = {2017}}

@article{Huang-etal18A,
Title = {A General Embedding Framework for Heterogeneous Information Learning in Large-Scale Networks},  
Author = {Xiao Huang and Jundong Li and Na Zou and Xia Hu},  
Booktitle = {ACM Transactions on Knowledge Discovery from Data},  
Volume = {12},  
Year = {2018}}
