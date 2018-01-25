function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
 test_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
 test_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
err = zeros(64,3);
k = 1;
for i=test_C
    for j=test_sigma
        % train madel on training data set
        model= svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
        % check result of model on cross validation set
        prediction = svmPredict(model, Xval);
        err(k,1) = i;
        err(k,2) = j;
        err(k,3) = mean(double(prediction ~= yval));
        k+=1;
    end;
end;

[m,idx] = min(err(:,3));
C = err(idx,1);
sigma = err(idx,2);
% C = 10;
% sigma = 0.03;
% error_min = inf;
% values = [0.01 0.03 0.1 0.3 1 3 10 30];

% for _C = values
%   for _sigma = values
%     fprintf('Train and evaluate (on cross validation set) for\n[_C, _sigma] = [%f %f]\n', _C, _sigma);
%     model = svmTrain(X, y, _C, @(x1, x2) gaussianKernel(x1, x2, _sigma));
%     e = mean(double(svmPredict(model, Xval) ~= yval));
%     fprintf('prediction error: %f\n', e);
%     if( e <= error_min )
%       fprintf('error_min updated!\n');
%       C = _C;
%       sigma = _sigma;
%       error_min = e;
%       fprintf('[C, sigma] = [%f %f]\n', C, sigma);
%     end
%     fprintf('--------\n');
%   end
% end
% =========================================================================

end
