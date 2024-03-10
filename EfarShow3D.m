clear;clc;
incident=[10,20,30,40,50,60,70];
ACorrelationCoefficient=zeros(2,length(incident));
ARMSE = zeros(2,length(incident));
vidObj=VideoWriter(['Video01','.mp4'],'MPEG-4');
vidObj.FrameRate = 1;
open(vidObj);
inciphi = 225;
for idx = 1:length(incident)
    keyinfo = ['Efar' num2str(inciphi) '_Zenith_' num2str(incident(idx)) '_VERIFY.mat'];
    Efar = load(['Efar/mat/' keyinfo]);
    REAL = Efar.Ereal; FAKE = Efar.Efake;

    [row, col] = size(REAL);
    IDX_R = zeros(row,col);
    IDX_F = zeros(row,col);
    
    inciangle = incident(idx);
    r1 = corrcoef(REAL,FAKE);
    threshold = max(max(FAKE))*0.05;
%     threshold = 0;
    REAL = max(REAL,threshold);FAKE = max(FAKE,threshold);
    IDX_R(REAL<threshold)=1;IDX_F(FAKE<threshold)=1;
    IDXD=IDX_R.*IDX_F;
    Ereal = 10*log10(REAL);Efake = 10*log10(FAKE);
    tmp = (Ereal-Efake).^2;
    db_rmse = sqrt(sum(tmp(:))/(row*col-sum(IDXD(:))));
   
    % 求主要区域功率比
    REAL_main = REAL(IDXD==1);
    a = sum(REAL_main.^2);
    b = sum(REAL(:).^2);
    c=a/b;

    db_rmse_Mainloab = sqrt(sum(tmp(:))/(row*col-sum(IDXD(:))));
%     pcolor(IDXD);
    
    ARMSE(1,idx) = db_rmse;ARMSE(2,idx) = db_rmse_Mainloab;
    ACorrelationCoefficient(1,idx) = r1(1,2);
    
    r = 1;num_theta = 91;num_phi= 361;
    theta = linspace(0,pi/2,num_theta);phi = linspace(0,2*pi,num_phi);
    [theta,phi] = meshgrid(theta,phi);
    x = r*sin(theta).*cos(phi);y = r*sin(theta).*sin(phi);z=r*cos(theta);
    
    fig = figure(1);
    set(fig,'Color','white','Renderer','openGL');
    set(gcf,'unit','normalized','position',[0.1,0.1,0.8,0.5]);
    tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
    azimuth = 50;
    nexttile
%     pcolor(REAL');
    surf(x, y, z, REAL');
    shading interp
    view(azimuth, 20);
    axis equal;  
    t2 = title('真实远场 Efar');axis tight; 
    set(t2, 'FontName', '微软雅黑', 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'XTick', []);  
    set(gca, 'YTick', []);  
    set(gca, 'ZTick', []);  
    xlabel('X');ylabel('Y');zlabel('Z');
    axis off;

    nexttile
%     pcolor(FAKE');
    surf(x, y, z, FAKE');
    shading interp
    view(azimuth, 20);
    axis equal;  
    t1 = title('重建远场 Efar');axis tight; 
    set(t1, 'FontName', '微软雅黑', 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'XTick', []); 
    set(gca, 'YTick', []); 
    set(gca, 'ZTick', []); 
    xlabel('X');ylabel('Y');zlabel('Z');
    colormap('jet');
    axis off;
    c1 = colorbar('Position', [0.93, 0.25, 0.015, 0.5]);  % [x, y, width, height]
    c2 = colorbar('Position', [0.5, 0.25, 0.015, 0.5]);  % [x, y, width, height]
    set(get(c1,'Title'),'string','E_{far} [V/m]','FontSize',10);
    set(get(c2,'Title'),'string','E_{far} [V/m]','FontSize',10);
 
    drawnow;
    pause(1);
    saveas(gcf,['Efar','\img\',keyinfo,'.jpg']);
    currentFrame = getframe(fig);
    writeVideo(vidObj,currentFrame);
    clf(fig,'reset');
end
close(vidObj);

