function [ut,auxt,auxt_labels]=regulator_DOCONT_v4_v2(i,t,x,y,u,ts,csi,o,aux)

% DOCONT rule-based control strategy - variant v4 
%
% [ut,auxt,auxt_labels]=regulator_DOCONT_v4(i,t,x,y,u,ts,csi,o,aux)
%
% OUTPUT
% ut ... row vector of values of manipulated variables at time t
% auxt ... row vector of auxiliary variables for optional storage of time-varying variables and parameters of control strategies (e.g. setpoints)
%
% ut(1) ... feed flowrate (F, ml/s)
% ut(2) ... air flowrate (Fv, ml/s)  
% ut(3) ... stirrer speed (n, 1/s)
% ut(4) ... acid flowrate (F_a, ml/s)
% ut(5) ... base flowrate (F_b, ml/s)

% auxt(1) ... auxiliary variable no. 1
% auxt(2) ... auxiliary variable no. 2
% auxt(3) ... auxiliary variable no. 3
% auxt(4) ... auxiliary variable no. 4
% auxt(5) ... auxiliary variable no. 5
%
% auxt_labels(1) ... text label for auxiliary variable no. 1
% auxt_labels(2) ... text label for auxiliary variable no. 2
% auxt_labels(3) ... text label for auxiliary variable no. 3
% auxt_labels(4) ... text label for auxiliary variable no. 4
% auxt_labels(5) ... text label for auxiliary variable no. 5
%
% INPUT
% i ... integer index value passed from the parent FOR loop
% t ... current time (s)
% x ... matrix of state variables
% y ... matrix of output (measured) variables
% u ... matrix of manipulated variables
% ts ... sample time (s)
% csi ... glucose concentration in the feed (mol/ml)
% o ... matrix of estimated variables by soft sensors
% aux ... matrix of auxiliary variables for optional storage of time-varying variables and parameters of control strategies (e.g. setpoints)
% Note: for the composition of the x,y,u,o,aux matrices, see help for yeast_simcon.m function

global citac_spusteni_reg
global t_inc
global dose

ut=zeros(1,5);
auxt=zeros(1,5)*NaN;
auxt_labels={'auxiliary variable no. 1','auxiliary variable no. 2','auxiliary variable no. 3','auxiliary variable no. 4','auxiliary variable no. 5'};


% air flowrate set to a constant value of 10 l/min, i.e. 167 ml/s
ut(2)=167;

% stirrer speed set to a constant value of 600 1/min, i.e. 10 1/s
ut(3)=10;

% acid flowrate set to a constant value of - ml/s
% ut(4)=0;

% base flowrate set to a constant value of - ml/s
% ut(5)=0;
% 
% if t>5000
%     ut(4)=0.005;
% end

% feed flowrate - initial value (ml/min)
dose_initial=0.6;

% feed flowrate - increment (ml/min)
dose_delta=0.3;

% feed flowrate - upper limit (ml/min)
dose_max=10;

if i==1
	t_inc=i;
	dose=dose_initial;
end





% DO change threshold (% saturation)
DO_delta=0.1;

DO_current=y(i,4);
if (i>1)
DO_previous=y(i-1,4);
else
DO_previous=DO_current;
end

DO_dif=DO_current-DO_previous;

if DO_dif>DO_delta
	gain=1.1;
	dose=min([gain*dose dose_max]);
	on=1;
else
	on=1;
end

ut(1)=on*dose/60;

citac_spusteni_reg=citac_spusteni_reg+1;


% safety check against overfilling of the bioreactor

% maximum total allowable volume of liquid in the bioreactor (ml)
% vol_max=7500;
% % 
% cur_vol=x(i,6);
% new_vol=cur_vol+ut(1)*60+u(5)*60+u(4)*60;
% 
% if new_vol>vol_max
% 	ut(1)=(vol_max-cur_vol)/60;
%     u(4)=0;
%     u(5)=0;
% end
% feed = ut(1);


% ph control
% if i==1
%     u(4)=0;
%     u(5)=0;
% end

% calculation of the setpoint for biomass concentration control (exponential growth with the desired biomass growth rate kw)
auxt(1)=5.5;

% calculation of control error (difference between setpoint and estimated biomass concentration value)
e=auxt(1)-y(i,5);

if e>0
    kP = 0.0001;
    auxt(2)=kP*e;
    % integration of the control error (trapezoidal method of numerical integration)
    global old_e_cum_base
    global old_e_base

    if isempty(old_e_cum_base)
	    old_e_cum_base=0;
    end

    if isempty(old_e_base)
	    old_e_base=e;	
    end

    delta_e_cum_base=ts*(e+old_e_base)/2;
    e_cum_base=delta_e_cum_base+old_e_cum_base;
    old_e_base=e;
    old_e_cum_base=e_cum_base;

    % gain value of the integral (I) component of the PI controller
    kI=0.000006;

    % output of the proportional (I) component of the PI controller
    auxt(3)=kI*e_cum_base;
    if t==0
        ut(5)=0;
    else

        ut(5)=(auxt(2)+auxt(3));
        ut(4)=0;
        base = ut(5)
    end
end

if e<0
    kP = -0.000036;
    auxt(2)=kP*e;
    % integration of the control error (trapezoidal method of numerical integration)
    global old_e_cum_acid
    global old_e_acid

    if isempty(old_e_cum_acid)
	    old_e_cum_acid=0;
    end

    if isempty(old_e_acid)
	    old_e_acid=e;	
    end

    delta_e_cum_acid=ts*(e+old_e_acid)/2;
    e_cum_acid=delta_e_cum_acid+old_e_cum_acid;
    old_e_acid=e;
    old_e_cum_acid=e_cum_acid;

    % gain value of the integral (I) component of the PI controller
    kI=-0.000011;

    % output of the proportional (I) component of the PI controller
    auxt(3)=kI*e_cum_acid;
    if t==0
        ut(4)=0;
    else

        ut(4)=auxt(2)+auxt(3);
        ut(5)=0;
        acid = ut(4)
    end
end
citac_spusteni_reg=citac_spusteni_reg+1;

vol_max=7500;
% 
cur_vol=x(i,6);
new_vol=cur_vol+ut(1)*60+u(5)*60+u(4)*60;

if new_vol>vol_max
	ut(1)=(vol_max-cur_vol)/60;
    ut(4)=0;
    ut(5)=0;
end
feed = ut(1)


%% this is the part before calculating the parameters

% if x(i,14)<0
%     x(i,14)
% end

% % gain value of the proportional (P) component of the PI controller
% kP=1;
% 
% % output of the proportional (P) component of the PI controller
% auxt(2)=kP*e;
% 
% % integration of the control error (trapezoidal method of numerical integration)
% global old_e_cum
% global old_e
% 
% if isempty(old_e_cum)
% 	old_e_cum=0;
% end
% 
% if isempty(old_e)
% 	old_e=e;	
% end
% 
% delta_e_cum=ts*(e+old_e)/2;
% e_cum=delta_e_cum+old_e_cum;
% old_e=e;
% old_e_cum=e_cum;
% 
% % gain value of the integral (I) component of the PI controller
% kI=1;
% 
% % output of the proportional (I) component of the PI controller
% auxt(3)=kI*e_cum;
% 
% % calculation of the feed flowrate
% % estimated feed demand (feedforward part) adjusted by the output of the PI controller (feedback part)
% 
% if e>0
%     ut(5)=0.005+auxt(2)+auxt(3);
%     ut(4)=0;
%     base = ut(5)
% end
% 
% if e<0
%     ut(4)=0.007+auxt(2)+auxt(3);
%     ut(5)=0;
%     acid = ut(4)
% end
% 
% citac_spusteni_reg=citac_spusteni_reg+1;



% safety check against overfilling of the bioreactor


