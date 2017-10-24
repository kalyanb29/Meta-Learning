function [f, g,cv,tfe] = srn(x,tfe)
	if nargin == 0
		prob.nx = 2;
		prob.nf = 2;
		prob.ng = 2;
		prob.range(1,:) = [-20,20];
		prob.range(2,:) = [-20,20];
		f = prob;
	else
		[f,g,cv,tfe] = srn_true(x,tfe);
	end
end


function [f,g,cv,tfe] = srn_true(x,tfe)
	f(:,1) = 2 + (x(:,1)-2).^2 + (x(:,2)-1).^2;
	f(:,2) = 9*x(:,1) - (x(:,2)-1).^2;

	g(:,1) = -x(:,1).^2 - x(:,2).^2 + 225;
	g(:,2) = -x(:,1) + 3*x(:,2) - 10;
    g = -g;
    cv = sum(max(g,0),2);
    tfe = tfe + size(x,1);
end
