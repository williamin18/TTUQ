clear variables

m = 3;
d = 29;
n_samples = 1000;


[training_idx,training_xi,n_samples] = genTensorCompletionSamples(m,d,n_samples,'Hermite');

[r1_sample_xi] = genRank1Samples(m,d,'Hermite');