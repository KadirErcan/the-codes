function y=sasb(h,x,p)

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

%% dissociation constants
p = p(1,:);

%% concentrations
cHCO3 = (p(42)*p(43))/h;
cCO3 = (p(42)*p(43)*p(44))/h^2;
%cCO2total = (1 + p(42) + p(42)*p(43)/h + p(42)*p(43)*p(44)/h^2)*cc;
%cEtOHtotal = (1+p(45)/h)*ce;

    % t = t0:ts:t1;
    % % Define time vector t and manipulated variable u(t)
    % 
    % % Integrate the manipulated variable using cumtrapz
    % VetNaOH = cumtrapz(t, u(5));  % Cumulative integral of u(t)
    % cnaoh = p(55)*VetNaOH/(vt+VetNaOH);
    % 
    % % Integrate the manipulated variable using cumtrapz
    % VetH2SO4 = cumtrapz(t, u(4));  % Cumulative integral of u(t)
    % ch2so4 = p(56)*VetH2SO4/(vt+VetH2SO4);


%% degree of dissociation
alphaC = 1 + p(42) + ((p(42)*p(43))/h) + (p(42)*p(43)*p(44))/h^2;
alphaE = 1 + p(45)/h;
%% charge balance
y = h - p(46)/h + (cnaoh-2*ch2so4) - (cHCO3/alphaC)*cc - 2*(cCO3/alphaC)*cc  - ((p(45)/h)/alphaE)*ce;

end

