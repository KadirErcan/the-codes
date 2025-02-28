function [sys, xinit, str, ts] = dist_sf_2_trial (t, x, u, flag)
%% parameters - system dimensions, model parameters


N  = 8;      % number of trays
nu = 4;  % inputs = [xF nF nD nV]
nx = N+2;  % number of states
ny = N+2;  % number of outputs

% model parameters

k = 5;       % feed tray number
nB = 1;      % reboiler hold up [kmol]
nT = 0.2;    % tray and condenser hold up [kmol]
effT = 0.80; % efficacy - tray
effB = 1.00; % efficacy - reboiler

% inputs
xF = 0.65;  % molar ratio of light component in feed
F  = 0.300; % molar flow of feed
D  = 0.179; % molar flow of distillate
V  = 0.328; % molar flow of vapour

%u = [0.65 0.300 0.179 0.300-0.179];

% methanol-Water params
a =   0.00046224365;
b =  15.131084;
c =  -5.1346083;
d =  25.2741;
e = -16.30502;
%% initial conditions x(0) = x0

% initial guess
%X0  = linspace(0,1,N+2);
%sol = fsolve(@(x) dynamics_equations(x,u), X0);

% xs = [0.7410    0.6868    0.6391    0.5912
%       0.5685    0.5239    0.4332    0.2512    0.0362]
%x0 = sol;
x0 = [0.9526 0.8926 0.8406 0.7916 0.7398 0.6703 0.6478 0.5953 0.4857 0.2023];
%x0=  [0.7518 ...
 % 0.4921
  %  0.2853
   % 0.1698
    %0.1307
    %0.1606
    %0.0401
    %0.0084
    %0.0014
   %-0.0002];%x0 = [0.7472    0.6970    0.6537    0.6141    0.5761    0.5482    0.4878    0.3635    0.1454    0.0152];

%% dynamics equations dx(t)/dt = f(x(t), u(t))

    function f = dynamics_equations (x, u)
        xF = u(1);
        F  = u(2);
        D  = u(3);
        B  = F-D;
        B  = u(4);
        L  = V-D; % molar flow of liquid;      condenser MB: nV = nL + nD
      % molar flow of bottom product; column MB: nF = nW + nD

        %% Annotation
        %
        %   N = 7
        %
        %   1         Total Condenser   0 (+ 1)
        %  -------------------------------
        %   2         First Tray        1
        %   3                           2
        %  -------------------------------
        %   4         Feed Tray         3
        %  -------------------------------
        %   5                           4  
        %   6                           5
        %   7                           6
        %   8 (+ 1)   Last Tray         7 
        %  -------------------------------
        %   9 (+ 2)   Reboiler          8 (+ 2)
        %  ================================
        %   9 = N+2                   (0:8)

        % function (x-y diagram)
        y_e = @(x) (-23.805*x^6+82.142*x^5-112.57*x^4+78.529*x^3-29.821*x^2+6.5244*x+0.003);
        %(-23.805*x^6+82.142*x^5-112.57*x^4+78.529*x^3-29.821*x^2+6.5244*x+0.003);(a+b*x+c*x^2)/(1+d*x+e*x^2);
        
        % calculate vapour molar fractions   (reboiler => condenser)
        y = zeros(1,N+2);  
        y(N+2) = effB * y_e(x(N+2));
        for i = N+1:-1:2
            y(i) = effT*y_e(x(i))+(1-effT)*y(i+1);
        end
        
        f = [];
        % condenser
        f = [f; 
             (V*y(2) - (L + D)*x(1)) / nT];
    
        % rectifying section
        for i = 2:k
            f = [f;
                 (V*y(i+1) + L*x(i-1) - ...
                  V*y(i) - L*x(i)) / nT];
        end

        % feed tray
        f = [f;
             (V*y(k+2)+F*xF+L*x(k)-...
             V*y(k+1)-(L+F)*x(k+1))/nT];

        % stripping section
        for i = k+2:N+1
            f = [f;
                 (V*y(i+1) + (L+F)*x(i-1) - ...
                  V*y(i) - (L+F)*x(i)) / nT];
        end

        % reboiler
        f = [f;
             ((L+F)*x(N+1) - B*x(N+2)-V*y(N+2)) / nB];

    end


%% output equations y(t) = g(x(t))

    function g = output_equations (x)
        g = x;
    end

%% END OF S-FUNCTION

% figure
% xx = linspace(0,1,1e3)
% yy = (a + b.*xx + c*xx.^2) ./ (1 + d.*xx + e*xx.^2)
% plot(xx,yy)
% hold on; grid on; box on; grid minor;
% plot([0 1],[0 1])
% title('x-y diagram (EtOH-H2O)')
% xlabel('(l) mol frac, x')
% ylabel('(g) mol frac, y')


%% other functions // do not change unless really needed
switch flag
    case 0
        [sys, xinit, str, ts] = mdlInitializeSizes(nx, ny, nu, x0);
    case 1
        sys = mdlDerivatives(t,x,u);
    case 3
        sys = mdlOutputs(t,x,u);
    case {2, 4, 9}
        sys = [];
end

    function [sys, xinit, str, ts] = mdlInitializeSizes(nx, ny, nu, x0)
        sizes = simsizes;
        sizes.NumSampleTimes = 1;
        sizes.NumContStates  = nx;
        sizes.NumOutputs     = ny; 
        sizes.NumInputs      = nu; 
        xinit = x0; 
        sys = simsizes(sizes);
        str = []; ts = [0 0];
    end
    function f = mdlDerivatives(~, x, u)
        f = dynamics_equations (x, u);
    end
    function y = mdlOutputs(~, x, ~)
        y = output_equations(x);
    end
end
