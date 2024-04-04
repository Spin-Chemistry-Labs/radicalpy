clear; close all; clc;

% Spin system
Sys.S = [1/2 1/2];
Sys.g = [2.0023; 2.0023];
% Sys.Nucs = '1H, 1H';
% Sys.A = [mt2mhz(0.4), 0; 0, mt2mhz(0.5)];
Sys.Nucs = '1H';
Sys.A = [mt2mhz(0.4), 0];
Sys.J = 0;  % MHz
Sys.dip = 0;  % MHz

Singlet = 1/sqrt(2)*[0; 1; -1; 0];  % singlet state in uncoupled basis, (|��>-|��>)/sqrt(2)
PSinglet = Singlet*Singlet';  % singlet projection operator
Singlet = kron(PSinglet, eye(2));
rho0 = Singlet / trace(Singlet);
B = [0, 0, 0];
ham = ham(Sys, B);
n = 600;
dt = 5e-3;

td = evolve(rho0, Singlet, ham, n, dt);
% writematrix(real(td), "easyspin_1nuc.txt")


time = linspace(0, 3e-6, n);
figure(1),clf(1)
hold on
plot(time, td)