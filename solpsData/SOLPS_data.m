%% 5/5/2023, Shahinul
%% 252*32 size; axial: 1-252 (0.5 -4.2 m), radial1-32 
%% Target at 4.2 m

load('C:\Users\17i\Desktop\GITRWallDYN\matlab_SOLPS.mat'); %%uploaded on github
pc = phys_const; fs =14; lw = 3;

Z = Case{1}.Geo.LcRight(:,1)+0.5;
B = Case{1}.Geo.bb(:,:,1);

%%%Plasma parameters 
ne = Case{1}.State.ne(:,:);  % m^-3
Te = Case{1}.State.te(:,:)./pc.eV;  % eV
Ti = Case{1}.State.ti(:,:)./pc.eV;   %eV
ui = -1.*Case{1}.State.ua(:,:,2); % m/s
ni = Case{1}.State.na(:,:,2) ; %m^-3


%%%%Neutral data on EIRENE grid 
indAtom = 1; 
n_atom = Case{1}.Trineuts.a.n(:,indAtom); 
n_mol = Case{1}.Trineuts.m.n(:,indAtom); 

%%%Neutral data on B2 grid 
n_atom_B2 =load('C:\Users\17i\Desktop\GITRWallDYN\dab2.mat'); %%uploaded on github
n_mol_B2 =load('C:\Users\17i\Desktop\GITRWallDYN\dmb2.mat');
T_atom_B2 = load('C:\Users\17i\Desktop\GITRWallDYN\tab2.mat');
T_mol_B2 = load('C:\Users\17i\Desktop\GITRWallDYN\tmb2.mat');


figure; hold on; box on; grid on;
plot(Z, Te(:,1),'-k', 'linewidth',2.0);
title('r =0 m')
ylabel ('T_e(eV)')
set(gca,'fontsize',fs)
set(gcf,'color','w') 

figure; hold on; box on; grid on;
plot(Z, ne(:,1),'-k', 'linewidth',2.0);
title('r =0 m')
ylabel ('n_e (m^{-3})')
set(gca,'fontsize',fs)
set(gcf,'color','w') 

figure; hold on; box on; grid on;
plot(Z, ui(:,1),'-k', 'linewidth',2.0);
title('r =0 m')
ylabel ('u_{i||} (m/s)')
set(gca,'fontsize',fs)
set(gcf,'color','w') 

figure; hold on; box on; grid on;
plot(Z, B(:,1),'-k', 'linewidth',2.0);
title('r =0 m')
ylabel ('B (T)')
set(gca,'fontsize',fs)
set(gcf,'color','w') 

figure; hold on; box on; grid on;
plot(Z, n_atom_B2.dab2(:,1),'-k', 'linewidth',2.0);
title('r =0 m')
ylabel ('n_{D^0} (m^{-3})')
set(gca,'fontsize',fs)
set(gcf,'color','w') 

figure; hold on; box on; grid on;
plot(Z, n_mol_B2.dmb2(:,1),'-k', 'linewidth',2.0);
title('r =0 m')
ylabel ('n_{D_2} (m^{-3})')
set(gca,'fontsize',fs)
set(gcf,'color','w') 

figure; hold on; box on; grid on;
plot(Z, T_atom_B2.tab2(:,1)./pc.eV,'-k', 'linewidth',2.0);
title('r =0 m')
ylabel ('T_{D^0} (eV)')
set(gca,'fontsize',fs)
set(gcf,'color','w') 