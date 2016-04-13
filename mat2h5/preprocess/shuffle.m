function Xs = shuffle(X)
Xs = X(randperm(size(X,1)), :);