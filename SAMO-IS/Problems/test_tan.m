function[f,g,cv,tfe] = test_tan(x,tfe)
if nargin == 0
	prob.nf = 3;
	prob.ng = 0;  
	prob.nx = 2;
    for i = 1:prob.nx
        prob.range(i,:) = [-3,3];
    end
	f = prob;
else
	[f,g,cv,tfe] = Test_Tan_true(x,tfe);
end
return


function [f,g,cv,tfe] = Test_Tan_true(x,tfe)
f(:,1) = 0.5.*(x(:,1).^2 + x(:,2).^2) + sin(x(:,1).^2 + x(:,2).^2);
f(:,2) = (3*x(:,1) - 2*x(:,2) + 4).^2/8 + (x(:,1) - x(:,2) + 1).^2/27 + 15;
f(:,3) = 1./(x(:,1).^2 + x(:,2).^2 + 1) - 1.1*exp(-x(:,1).^2 - x(:,2).^2);
g = [];
cv = zeros(size(x,1),1);
tfe = tfe + size(x,1);
return