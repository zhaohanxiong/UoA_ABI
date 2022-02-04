function dxdt = derivsecgsynV2(t,x,flag,rr,sfint,thetai,ai,bi,fhi)

% computes dxdt = F(t,x) taking input: 
%	t		= time steps
%	x		= initial conditions for ODEs [x y P C T]
% 	rr 		= rr process 
% 	sfint 	= Internal sampling frequency [Hertz]
% 	thetai 	= angles of extrema [radians] (also called theta)
% 	ai 		= z-position of extrema 
% 	bi 		= Gaussian width of peaks 
%   fhi     = respiratory rate
%
% Outputs:
%	dxdt 	= solutions [x' y' P' C' T']

% 1st/2nd ODEs
gamma 	= 1.0 - sqrt(x(1)^2 + x(2)^2);
w 		= 2*pi./rr(1 + floor(t*sfint));

dx1dt = gamma*x(1) - w*x(2);
dx2dt = gamma*x(2) + w*x(1);

% 3rd-5th ODEs
dtheta 	= rem(atan2(x(2),x(1)) - thetai, 2*pi);
PCT0 	= 0.05*sin(2*pi*fhi*t);

dx3dt = - sum((ai(1:2).*w./(bi(1:2).^2)).*dtheta(1:2).*exp(-0.5*(dtheta(1:2)./bi(1:2)).^2)) - (x(3) - PCT0);
dx4dt = - sum((ai(3:5).*w./(bi(3:5).^2)).*dtheta(3:5).*exp(-0.5*(dtheta(3:5)./bi(3:5)).^2)) - (x(4) - PCT0);
dx5dt = - sum((ai(6:7).*w./(bi(6:7).^2)).*dtheta(6:7).*exp(-0.5*(dtheta(6:7)./bi(6:7)).^2)) - (x(5) - PCT0);

% output
dxdt = [dx1dt; dx2dt; dx3dt; dx4dt; dx5dt];