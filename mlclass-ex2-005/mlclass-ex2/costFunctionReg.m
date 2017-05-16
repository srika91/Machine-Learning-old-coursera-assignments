function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h=X*theta;
q=sigmoid(h);
qq=y.*log(q);
qqq=(1-y).*log(1-q);
grad(1)=((1/m)*sum((q-y).*X(:,1)));
theta(1,:)=[];
J=((1/(m))*sum(-qq-qqq))+(lambda/(2*(m)))*sum(theta.^2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================


X(:,1)=[];
[a,b]=size(theta);
 for i=1:a
grad(i+1)=((1/m)*sum((q-y).*X(:,i)))+(lambda/m)*theta(i);
end



end
