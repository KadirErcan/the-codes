function yout=f(t,x,u,p)

% Function for calculating the vector of the right sides of the state equations  dx/dt=f(t,x,u,p) according to the well-known theory of limited respiratory capacity developed by (Sonnleitner and Käppeli, 1986) and (Sonnleitner and Hahnemann, 1994)
%
% yout=f(t,x,u,p)
%
% yout...row vector of derivatives of state variables with respect to time dx/dt
% t...current time (s)
% x...row vector of state variables
% u...row vector of manipulated variables
% p...row vector of system parameters

yout=zeros(1,length(x));

fun = @(h) wawb(h,x,p);
z = fzero(fun,[1e-20;100000]);
ch_ion=z; % mol/ml
%ch_ion=ch_ion*1000; %mol/l

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



qo=qocap*co/(co+p(26))*p(38)/(ce+p(38));
qs=p(21)*cs/(cs+p(24));
qe=p(22)*ce/(ce+p(25))*p(37)/(cs+p(37));

qs1=min([qo/(1+1.05*p(27)) qs]);
qs2=qs-qs1;
qe3=min([(qo-(1+1.05*p(27))*qs1)/(1.5+1.05*p(29)) qe]);

rs1=-qs1*cx;  
rx1=p(27)*rs1;
ro1=(1+1.05*p(27))*rs1;
rc1=(-1-p(27))*rs1;
ra1=p(57)*rs1;

rs2=-qs2*cx;   
rx2=p(28)*rs2;
re2=(-2/3-0.7*p(28))*rs2;
rc2=(-1/3-0.3*p(28))*rs2;
ra2=p(58)*rs2;

re3=-qe3*cx;   
rx3=p(29)*re3;
ro3=(1.5+1.05*p(29))*re3;
rc3=(-1-p(29))*re3;
ra3=p(59)*re3;

rx=rx1+rx2+rx3;
rs=rs1+rs2;
re=re2+re3;
ro=ro1+ro3;
rc=rc1+rc2+rc3;
ra=ra1+ra2+ra3;


mee=rx/cx;

indic=-help-sup;

if indic>=0
	kon1=1/p(40);
else
	kon1=mee;
end

qoul=p(23)*min([max([0 1+sup+help]) 1]);
qodes=(1+1.05*p(27))*qs+(1.5+1.05*p(29))*qe;

if qocap>=qodes & qocap<=qoul
	kon2=mee;
elseif qocap>=qodes & qocap>qoul
	kon2=1/p(39);
else
	kon2=1/p(41);
end

akla=0.026*((p(12)*1e+6)/vt)^0.4*p(11)^0.5;


% how to find Henry's constant for ammonia
cew=(u(2)*p(18)/vt+akla*ce)/(u(2)*p(2)*p(3)/(vt*p(1)*p(10))+akla);
cow=(u(2)*p(19)/vt+akla*co)/(u(2)*p(2)*p(4)/(vt*p(1)*p(10))+akla);
ccw=(u(2)*p(20)/vt+akla*cc)/(u(2)*p(2)*p(5)/(vt*p(1)*p(10))+akla);
caw=(u(2)*p(60)/vt+akla*ca)/(u(2)*p(2)*p(61)/(vt*p(1)*p(10))+akla);

%%
vvi=u(1);
F_a = u(4);
F_b = u(5);

F_in = vvi + F_a + F_b;
Kalvol=137.706;
Kw = 1e-14; %mol^2/l^2
Kal = 2e-5; %mol/l

%%
%yout(1)=rx+(F_in)/vt*(p(13)-cx);
% yout(1)=0;
% yout(2)=0;
% yout(3)=0;
% yout(4)=0;
% yout(5)=0;

yout(1)=rx+(vvi*p(13)/vt)-(F_in*cx/vt);
yout(2)=rs+(vvi*p(14)/vt)-(F_in*cs/vt); % changed
%yout(3)=re+(F_in)/vt*(p(15)-ce)+akla*(cew-ce);
yout(3)=re+(vvi*p(15)/vt)-(F_in*ce/vt)+akla*(cew-ce);
yout(4)=ro+(vvi*p(16)/vt)-(F_in*co/vt)+akla*(cow-co); % changed
%yout(5)=rc+(F_in)/vt*(p(17)-cc)+akla*(ccw-cc);
yout(5)=rc+(vvi*p(17)/vt)-(F_in*cc/vt)+akla*(ccw-cc);
yout(6)=F_in;
yout(7)=kon2*(-qocap+max([min([qodes qoul]) p(36)])); % min max
yout(8)=kon1*(-help-sup);
yout(9)=1/p(39)*(p(38)/(p(38)+ce)-1-sup);

yout(10)=F_b/vt*p(51) - F_in*cnaoh/vt;
% yout(10)=0;
yout(11)=F_a/vt*p(52) - F_in*ch2so4/vt;
% yout(11)=0;
yout(12)=F_b/vt*p(63) - F_in*cnh4oh/vt;
yout(13)=F_a/vt*p(53) - F_in*ch3po4/vt;
yout(14)=ra+(F_b*p(65)/vt)-(F_in*ca/vt) + (-Kalvol*u(2)/vt)*(Kw*ca/(Kal*ch_ion+Kw)); % changed
% 
% yout(12)=0;
% yout(13)=0;
% yout(14)=0;





