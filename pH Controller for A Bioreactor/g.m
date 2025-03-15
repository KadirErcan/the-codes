function yout=g(x,u,p)

% Function to calculate the vector of the right sides of the output equations  y=g(x,u,p)       
%
% yout=g(x,u,p)
%
% yout...row vector of output (measured) variables
%
% x...row vector of state variables
% u...row vector of manipulated variables
% p...row vector of system parameters

yout=zeros(1,5);

cx=x(1);
cs=x(2);
ce=x(3);
co=x(4);
cc=x(5);
vt=x(6);
qocap=x(7);
help=x(8);
sup=x(9);
cnaoh = x(10);
ch2so4 = x(11);
cnh4oh = x(12);
ch3po4 = x(13);
ca = x(14);

p = p(1,:);


akla=0.026*((p(12)*1e+6)/vt)^0.4*p(11)^0.5;

cew=(u(2)*p(18)/vt+akla*ce)/(u(2)*p(2)*p(3)/(vt*p(1)*p(10))+akla);
cow=(u(2)*p(19)/vt+akla*co)/(u(2)*p(2)*p(4)/(vt*p(1)*p(10))+akla);
ccw=(u(2)*p(20)/vt+akla*cc)/(u(2)*p(2)*p(5)/(vt*p(1)*p(10))+akla);



% xe (%)
yout(1)=ce*p(30)/p(31)*100;

% xoog (%)
yout(2)=cow*p(4)/(p(1)*p(10))*100;

% xcog (%)
yout(3)=ccw*p(5)/(p(1)*p(10))*100;

% DO (%)
yout(4)=co/cow*100;

% pH
fun = @(h) wawb(h,x,p);
z = fzero(fun,[1e-20;100000]);
yout(5)=-log10(z);
ph_values=yout(5)

