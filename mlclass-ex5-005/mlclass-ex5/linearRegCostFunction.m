function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples



% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
theta2=theta(2:end,:);
for i=1:m
h(i)=X(i,:)*theta;
end
a=sum((h'-y).^2)/(2*m);
a1=(sum(theta2.^2)*lambda)/(2*m);
J=a+a1;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

theta(1,:)=[];

grad(1)=((1/m)*sum((h'-y).*X(:,1)));
X(:,1)=[];
[a,b]=size(theta);
 for i=1:a
grad(i+1)=((1/m)*sum((h'-y).*X(:,i)))+(lambda/m)*theta(i);
end










% =========================================================================

grad = grad(:);

end
