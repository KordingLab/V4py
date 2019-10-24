function b = glmnet_boot(X, Y)
opts = glmnetSet;
opts.alpha = 0.1; opts.lambda = 0.1;

fit = glmnet(X, Y, 'poisson', opts);
b = [fit.a0; fit.beta];