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
% Find end-cap indices
cap1 = find(Zsurface ==0 & z1 < 0.50001 & z2 < 0.50001 & z3 < 0.50001);
cap2 = find(Zsurface ==0 & z1 > 4.139999 & z2 > 4.139999 & z3 > 4.139999);
subset = find(Zsurface >0);

% Calculate some stuff about centroids
% This script uses centroid locations to group surfaces
centroid = [1/3*(x1+x2+x3)', ...
    1/3*(y1+y2+y3)', ...
    1/3*(z1+z2+z3)'];

theta_centroid = atan2(centroid(:,2),centroid(:,1));
r_centroid = sqrt(centroid(:,2).^2 + centroid(:,1).^2);

% Inputs
nTheta = 6; %12; % Number of segments in theta
nZ = 15; %30; %15; % Number of axial segments

nTarget_radii = 4; % Number of radial locations at target
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

