function y=wawb(h,x,p)

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
%% %% dissociation constants
p = p(1,:);

%% concentrations for phosphoric acid dissociations

cH2PO4 = p(47)/h;
cHPO4 = (p(48)*p(47))/h^2;
cPO4 = (p(49)*p(48)*p(47))/h^3;
%cH3PO4 = (h^3*cPO4/(p(49)*p(48)*p(47)));
%cH3PO4total = cH3PO4*(1+p(47)/h+p(47)*p(48)/h^2+p(49)*p(48)*p(47)/h^3);
%% concentrations for ammonium hydroxide

cNH4 = (p(50)*h)/p(46);

%% concentrations for carbonic acid dissociation

cHCO3 = (p(42)*p(43))/h;
cCO3 = (p(42)*p(43)*p(44))/h^2;

%% alpha for carbonic acid
pseudoalphaC = 1+p(42)+((p(42)*p(43))/h)+(p(42)*p(43)*p(44))/h^2;

%% alpha for ethanol
pseudoalphaE = 1 + p(45)/h;

%% alpha for phosphoric acid

pseudoalphaP = 1 + p(47)/h + (p(48)*p(47))/h^2 + (p(49)*p(48)*p(47))/h^3;

%% alpha for ammonium hydroxide

pseudoalphaA = 1 + (p(50)*h)/p(46);

%% charge balance

y = h - p(46)/h ...
      + (cNH4/pseudoalphaA)*cnh4oh ...
      - (cH2PO4/pseudoalphaP)*ch3po4 - 2*(cHPO4/pseudoalphaP)*ch3po4 - 3*(cPO4/pseudoalphaP)*ch3po4 ...
      - (cHCO3/pseudoalphaC)*cc - 2*(cCO3/pseudoalphaC)*cc  ...
      - ((p(45)/h)/pseudoalphaE)*ce;
  

end

