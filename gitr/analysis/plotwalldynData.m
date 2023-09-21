clear all
close all
% Read in GITR geometry file
fid = fopen('../../walldyn3/data/gitrGeometryPointPlane3d.cfg');
tline = fgetl(fid);
tline = fgetl(fid);
for i=1:18
    tline = fgetl(fid);
    evalc(tline);
end
% Read in surface index groupings.
nSurfaces = 94;
for i=1:nSurfaces
    surf_ind_cell{i} = readmatrix(strcat('../../walldyn3/data/surface/surface_inds_',string(i)));
end
% For plotting different sections
colors = {'b','y','g','c','m','y'};
% Plot the complete GITR geometry (all the same color)
subset = 1:length(x1);

density_data_final = readmatrix('../../walldyn3/data/results/ProtoEmpex_ppext_NetAdensChange_100.000.dat'); 
% density_data_initial = readmatrix('../../walldyn3/data/results/ProtoEmpex_ppext_NetAdensChange_0.000.dat');

al_dens = abs(density_data_final(:, 2)); %- density_data_initial(:, 2);
n_dens = abs(density_data_final(:, 3)); %- density_data_initial(:, 3);
w_dens = abs(density_data_final(:, 4)); %- density_data_initial(:, 4);

% Read density data
% Read density data
% al_dens = density_data(:, 2);
% n_dens = density_data(:, 3);
% w_dens = density_data(:, 4);

% Create a new figure
figure;
colormap(parula);

densities = {al_dens, n_dens, w_dens};
titles = {'Al Areal Density Change', 'N Density Change', 'W Density Change'};
filename = {'al_density.png', 'n_density.png', 'w_density.png'};

for idx = 1:3
    subplot(1, 3, idx);
    hold on;
    for i = 1:nSurfaces
        subset = surf_ind_cell{i};
        color_value = densities{idx}(i);  % Use the appropriate density data
        plot_geom(subset, color_value, x1, x2, x3, y1, y2, y3, z1, z2, z3);
    end
    colorbar;
    title(titles{idx});
    xlabel('X [m]')
    ylabel('Y [m]')
    hold off;
end

% Save the entire figure as a PNG file
saveas(gcf, 'all_densities.png');


function plot_geom(subset, color_value, x1, x2, x3, y1, y2, y3, z1, z2, z3)
    X = [transpose(x1(subset)), transpose(x2(subset)), transpose(x3(subset))];
    Y = [transpose(y1(subset)), transpose(y2(subset)), transpose(y3(subset))];
    Z = [transpose(z1(subset)), transpose(z2(subset)), transpose(z3(subset))];
    
    patch(transpose(X), transpose(Y), transpose(Z), color_value, 'FaceAlpha', .3, 'EdgeAlpha', 0.3);
end