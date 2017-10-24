function [f, g,cv,tfe] = osy(x,tfe)
	if nargin == 0
		prob.nx = 6;
		prob.nf = 2;
		prob.ng = 6;
		prob.range(1,:) = [0,10];
		prob.range(2,:) = [0,10];
		prob.range(3,:) = [1,5];
		prob.range(4,:) = [0,6];
		prob.range(5,:) = [1,5];
		prob.range(6,:) = [0,10];
		f = prob;
	else
		[f,g,cv,tfe] = osy_true(x,tfe);
	end
end


function [f,g,cv,tfe] = osy_true(x,tfe)
	f(:,1) = -( 25*(x(:,1)-2).^2 + (x(:,2)-2).^2 + (x(:,3)-1).^2 + (x(:,4)-4).^2 + (x(:,5)-1).^2 );
	f(:,2) = x(:,1).^2 + x(:,2).^2 + x(:,3).^2 + x(:,4).^2 + x(:,5).^2 + x(:,6).^2;

	g(:,1) = x(:,1) + x(:,2) - 2;
	g(:,2) = 6 - x(:,1) - x(:,2);
	g(:,3) = 2 - x(:,2) + x(:,1);
	g(:,4) = 2 - x(:,1) + 3*x(:,2);
	g(:,5) = 4 - (x(:,3)-3).^2 - x(:,4);
	g(:,6) = (x(:,5)-3).^2 + x(:,6) - 4;
    g = -g;
    cv = sum(max(g,0),2);
    tfe = tfe + size(x,1);
end
