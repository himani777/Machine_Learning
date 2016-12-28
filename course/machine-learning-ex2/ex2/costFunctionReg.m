function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
t=X(:,2);
p=X(:,3);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


K=sigmoid(X*theta);

J=(-1/m)*sum(y.*log(K) + (1 - y).*log(1 - K));
reg=(lambda*(sum(theta(2:end))))/(2*m);
J=J+reg;



for i=1:m

grad=grad+(1/m)*(K(i)-y(i))*X(i,:)';

endfor


reg=lambda/m*[0;theta(2:end)];
grad=grad+reg;


% =============================================================

end
