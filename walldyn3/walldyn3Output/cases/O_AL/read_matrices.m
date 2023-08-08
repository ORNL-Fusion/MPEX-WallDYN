close all
clear all

% Matrices only had charges 0 through 3+ for both O and Al
n_charge_states = 4;

% Number of WALLDYN surfaces
nWd = 94;

names = {'O','Al'};

walldyn_matrices = zeros(94,94,n_charge_states,length(names));

for i=1:length(names)
    for j = 1:n_charge_states
        walldyn_matrices(:,:,j,i) = readmatrix(strcat('matrices/mat',names{i},string(j-1),'+.dat'));
        figure
        pcolor(1:1:nWd,nWd:-1:1,flipud(walldyn_matrices(:,:,j,i)))
        axis ij

        axis square
        colorbar
        ylabel('From')
        xlabel('To')
        title({'GITR Transfer Matrix',strcat('For WallDyn ',names{i},string(j-1),'+')})
    end
end