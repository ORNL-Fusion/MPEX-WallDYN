close all
clear all

% Read in 3D surface mesh geometry file
if (exist('x1') == 0)
    fid = fopen(strcat(pwd,'/gitrGeometryPointPlane3d.cfg'));

    tline = fgetl(fid);
    tline = fgetl(fid);
    for i=1:24
        tline = fgetl(fid);
        evalc(tline);
    end
    Zsurface = Z;
end
abcd = [a' b' c' d'];
surface = find(Zsurface);
nSurfaces = length(a);

subset = 1:length(x1);

figure
X = [transpose(x1(subset)),transpose(x2(subset)),transpose(x3(subset))];
Y = [transpose(y1(subset)),transpose(y2(subset)),transpose(y3(subset))];
Z = [transpose(z1(subset)),transpose(z2(subset)),transpose(z3(subset))];
patch(transpose(X),transpose(Y),transpose(Z),zeros(1,length(subset)),'FaceAlpha',.3,'EdgeAlpha', 0.3)%,impacts(surface)
title('Geometry')
xlabel('X [m]')
ylabel('Y [m]')
zlabel('Z [m]')
hold on;
% Find end-cap indices
cap1 = find(Zsurface ==0 & z1 < 0.50001 & z2 < 0.50001 & z3 < 0.50001);
cap2 = find(Zsurface ==0 & z1 > 4.139999 & z2 > 4.139999 & z3 > 4.139999);
subset = cap2;

figure
X = [transpose(x1(subset)),transpose(x2(subset)),transpose(x3(subset))];
Y = [transpose(y1(subset)),transpose(y2(subset)),transpose(y3(subset))];
Z = [transpose(z1(subset)),transpose(z2(subset)),transpose(z3(subset))];
patch(transpose(X),transpose(Y),transpose(Z),zeros(1,length(subset)),'FaceAlpha',.3,'EdgeAlpha', 0.3)%,impacts(surface)
title('Geometry')
xlabel('X [m]')
ylabel('Y [m]')
zlabel('Z [m]')

subset = find(Zsurface >0);

figure
X = [transpose(x1(subset)),transpose(x2(subset)),transpose(x3(subset))];
Y = [transpose(y1(subset)),transpose(y2(subset)),transpose(y3(subset))];
Z = [transpose(z1(subset)),transpose(z2(subset)),transpose(z3(subset))];
patch(transpose(X),transpose(Y),transpose(Z),zeros(1,length(subset)),'FaceAlpha',.3,'EdgeAlpha', 0.3)%,impacts(surface)
title('Geometry')
xlabel('X [m]')
ylabel('Y [m]')
zlabel('Z [m]')
hold on;

% Calculate some stuff about centroids
% This script uses centroid locations to group surfaces
centroid = [1/3*(x1+x2+x3)', ...
    1/3*(y1+y2+y3)', ...
    1/3*(z1+z2+z3)'];

theta_centroid = atan2(centroid(:,2),centroid(:,1));

r_centroid = sqrt(centroid(:,2).^2 + centroid(:,1).^2);


% Inputs
nTheta = 12; % Number of segments in theta
nZ = 15; % Number of axial segments

nTarget_radii = 6; % Number of radial locations at target
max_radius = 0.0614;

ind_cell = {};
zlim0 = 1.55; % sets lower axial bound on geometry
zlim1 = 1.95; % sets upper axial bound on geometry
thetalim0 = -pi;
thetalim1 = pi;

z_edges = zeros(nZ,2);
theta_edges = zeros(nTheta,2);

dz = (zlim1 - zlim0)/nZ;
zgrid = dz*(0:1:nZ) + zlim0;

z_edges(1,:) = [-100, zgrid(2)];
z_edges(end,:) = [zgrid(end-1), 100];
z_edges(2:end-1,1) = zgrid(2:end-2);
z_edges(2:end-1,2) = zgrid(3:end-1);

dtheta = (thetalim1 - thetalim0)/nTheta;
thetagrid = dtheta*(0:1:nTheta) + thetalim0;
theta_edges(:,1) = thetagrid(1:end-1);
theta_edges(:,2) = thetagrid(2:end);

dr = (max_radius)/nTarget_radii;
rgrid = dr*(0:1:nTarget_radii);
r_edges = zeros(nTarget_radii,2);
r_edges(:,1) = rgrid(1:end-1);
r_edges(:,2) = rgrid(2:end);

for i=1:nTheta
    for j=1:nZ
        ind_cell{(i-1)*nZ + j} = find( (Zsurface' >0) & (centroid(:,3)>= z_edges(j,1)) & (centroid(:,3) < z_edges(j,2)) ...
            & (theta_centroid >= theta_edges(i,1)) & (theta_centroid < theta_edges(i,2)));
    end
end
for i=1:nTarget_radii
    ind_cell{nTheta*nZ + i} = find(Zsurface ==0 & z1 > 4.139999 & z2 > 4.139999 & z3 > 4.139999 & r_centroid' >= r_edges(i,1) & r_centroid' < r_edges(i,2))
end
figure
colors = {'r','g','b','c','m','y'}
hold on
for i=1:nTheta*nZ + nTarget_radii
    subset = ind_cell{i};


    X = [transpose(x1(subset)),transpose(x2(subset)),transpose(x3(subset))];
    Y = [transpose(y1(subset)),transpose(y2(subset)),transpose(y3(subset))];
    Z = [transpose(z1(subset)),transpose(z2(subset)),transpose(z3(subset))];
    patch(transpose(X),transpose(Y),transpose(Z),colors{mod(i,6)+1},'FaceAlpha',1,'EdgeAlpha', 0.1); %,'FaceColor',colors{mod(i,6)})%,impacts(surface)
    title('Geometry')
    xlabel('X [m]')
    ylabel('Y [m]')
    zlabel('Z [m]')
end
hold on;
save('ind_cell.mat','ind_cell');

nP = 10; %1e6;
names = {"O","Al"};

masses = [16, 27];
for k=2:length(masses)
    for ii =1: nTheta*nZ+nTarget_radii
        areas = area(ind_cell{ii});
        area_cdf = cumsum(areas);
        area_cdf = area_cdf./area_cdf(end);
        area_cdf = [0; area_cdf'];

        x_sample = [];
        y_sample = [];
        z_sample = [];
        vx_sample = [];
        vy_sample = [];
        vz_sample = [];
        m = 27;
        nPoints = 200;
        maxE = 20;
        Eb = 3.39;%8.79;
        a = 5;
        E = linspace(0,maxE,nPoints);
        dE = E(2);
        thompson2 = a*(a-1)*E.*Eb^(a-1)./(E+Eb).^(a+1);
        xlabel('X [m]')
        ylabel('Y [m]')
        zlabel('Z [m]')
        title('Sample Particle Positions')

        m=masses(k);
        ecdf = cumsum(thompson2);
        ecdf = ecdf./ecdf(end);
        rand1 = rand(1,nP);
        randTheta = 2*pi*rand(1,nP);
        randPhi = 0.5*pi*rand(1,nP);
        Esamp = interp1(ecdf,E,rand1);

        v = sqrt(2*Esamp*1.602e-19/m/1.66e-27)';

        vx = v'.*cos(randTheta).*sin(randPhi);
        vy = v'.*sin(randTheta).*sin(randPhi);
        vz = v'.*cos(randPhi);

        if (length(find(vz<0)))
            negative_vz
        end

        buffer = 1e-5;
        plotSet = 1:length(area);
        planes = [x1' y1' z1' x2' y2' z2' x3' y3' z3'];

        X = [planes((plotSet),1),planes((plotSet),4),planes((plotSet),7)];
        Y = [planes((plotSet),2),planes((plotSet),5),planes((plotSet),8)];
        Z = [planes((plotSet),3),planes((plotSet),6),planes((plotSet),9)];

        nP0 = 0;
        nTriangles = length(planes);

        r1 = rand(nP,1);
        this_triangle = floor(interp1(area_cdf,1:length(areas)+1,r1));
        for j=1:nP

            i = ind_cell{ii}(this_triangle(j));
            x_tri = X(i,:);
            y_tri = Y(i,:);
            z_tri = Z(i,:);
            parVec = [x_tri(2) - x_tri(1), y_tri(2) - y_tri(1) , z_tri(2) - z_tri(1)];
            parVec = parVec./norm(parVec);
            samples = sample_triangle(x_tri,y_tri,z_tri,1);
            fprintf('x_tri point: (%f, %f, %f)\n', x_tri(1), x_tri(2), x_tri(3));

            fprintf('Sampled point: (%f, %f, %f)\n', samples(1), samples(2), samples(3));

     
            normal = inDir(i)*(-abcd(i,1:3)./plane_norm(i));

            v_inds =j; % nP0+1:nP0+nP;

            x_sample(v_inds) = samples(:,1) + buffer*normal(1);
            y_sample(v_inds) = samples(:,2) + buffer*normal(2);
            z_sample(v_inds) = samples(:,3) + buffer*normal(3);

            parVec2 = cross(parVec,normal);

            newV = vx(v_inds)'.*parVec + vy(v_inds)'.*parVec2 + vz(v_inds)'.*normal;
            vx_sample(v_inds) = newV(:,1);
            vy_sample(v_inds) = newV(:,2);
            vz_sample(v_inds) = newV(:,3);

        end
        ncid = netcdf.create(strcat('particle_sources/particle_source_',names{k},'_',string(ii),'.nc'),'NC_WRITE')

        dimP = netcdf.defDim(ncid,'nP',nP);

        xVar = netcdf.defVar(ncid,'x','double',[dimP]);
        yVar = netcdf.defVar(ncid,'y','double',[dimP]);
        zVar = netcdf.defVar(ncid,'z','double',[dimP]);
        vxVar = netcdf.defVar(ncid,'vx','double',[dimP]);
        vyVar = netcdf.defVar(ncid,'vy','double',[dimP]);
        vzVar = netcdf.defVar(ncid,'vz','double',[dimP]);

        netcdf.endDef(ncid);

        netcdf.putVar(ncid, xVar, x_sample);
        netcdf.putVar(ncid, yVar, y_sample);
        netcdf.putVar(ncid, zVar, z_sample);
        netcdf.putVar(ncid, vxVar, vx_sample);
        netcdf.putVar(ncid, vyVar, vy_sample);
        netcdf.putVar(ncid, vzVar, vz_sample);

        netcdf.close(ncid);
    end
end
axis equal

function samples = sample_triangle(x,y,z,nP)
x_transform = x - x(1);
y_transform = y - y(1);
z_transform = z - z(1);

v1 = [x_transform(2) y_transform(2) z_transform(2)];
v2 = [x_transform(3) y_transform(3) z_transform(3)];
v12 = v2 - v1;
normalVec = cross(v1,v2);

a1 = rand(nP,1);
a2 = rand(nP,1);

samples = a1.*v1 + a2.*v2;


samples2x = samples(:,1) - v2(1);
samples2y = samples(:,2) - v2(2);
samples2z = samples(:,3) - v2(3);
samples12x = samples(:,1) - v1(1);
samples12y = samples(:,2) - v1(2);
samples12z = samples(:,3) - v1(3);
v1Cross = [(v1(2).*samples(:,3) - v1(3).*samples(:,2)) (v1(3).*samples(:,1) - v1(1).*samples(:,3)) (v1(1).*samples(:,2) - v1(2).*samples(:,1))];
v2 = -v2;
v2Cross = [(v2(2).*samples2z - v2(3).*samples2y) (v2(3).*samples2x - v2(1).*samples2z) (v2(1).*samples2y - v2(2).*samples2x)];
v12Cross = [(v12(2).*samples12z - v12(3).*samples12y) (v12(3).*samples12x - v12(1).*samples12z) (v12(1).*samples12y - v12(2).*samples12x)];

v1CD = normalVec(1)*v1Cross(:,1) + normalVec(2)*v1Cross(:,2) + normalVec(3)*v1Cross(:,3);
v2CD = normalVec(1)*v2Cross(:,1) + normalVec(2)*v2Cross(:,2) + normalVec(3)*v2Cross(:,3);
v12CD = normalVec(1)*v12Cross(:,1) + normalVec(2)*v12Cross(:,2) + normalVec(3)*v12Cross(:,3);

inside = abs(sign(v1CD) + sign(v2CD) + sign(v12CD));
insideInd = find(inside ==3);
notInsideInd = find(inside ~=3);

fprintf('Indices: (%d, %d, %d)\n', inside, insideInd, notInsideInd);


v2 = -v2;
dAlongV1 = v1(1).*samples(notInsideInd,1) + v1(2).*samples(notInsideInd,2) + v1(3).*samples(notInsideInd,3);
dAlongV2 = v2(1).*samples(notInsideInd,1) + v2(2).*samples(notInsideInd,2) + v2(3).*samples(notInsideInd,3);

dV1 = norm(v1);
dV2 = norm(v2);
halfdV1 = 0.5*dV1;
halfdV2 = 0.5*dV2;

samples(notInsideInd,:) = [-(samples(notInsideInd,1) - 0.5*v1(1))+0.5*v1(1) ...
    -(samples(notInsideInd,2) - 0.5*v1(2))+0.5*v1(2) ...
    -(samples(notInsideInd,3) - 0.5*v1(3))+0.5*v1(3)];

samples(notInsideInd,:) = [(samples(notInsideInd,1) + v2(1)) ...
    (samples(notInsideInd,2) + v2(2)) ...
    (samples(notInsideInd,3) + v2(3))];


samples(:,1) = samples(:,1)+ x(1);
samples(:,2) = samples(:,2)+ y(1);
samples(:,3) = samples(:,3)+ z(1);

end
