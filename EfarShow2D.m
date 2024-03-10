clear;clc;
incident=[10,20,30,40,50,60,70];
ACorrelationCoefficient=zeros(2,length(incident));
ARMSE = zeros(2,length(incident));
vidObj=VideoWriter(['Efar_RS30_55_065_2','.mp4'],'MPEG-4');
vidObj.FrameRate = 1;
open(vidObj);
for idx = 1:length(incident)
    keyinfo = ['Efar180_' num2str(incident(idx)) '.mat'];
    Efar = load(['Efar2D/mat/' keyinfo]);
    REAL = [flip(Efar.Ereal(:,1));Efar.Ereal(:,181)];
    FAKE = [flip(Efar.Efake(:,1));Efar.Efake(:,181)];
    
    x = linspace(1,181,181);
    figure;
    plot(x,Efar.Ereal);
    hold on
    plot(x,Efar.Efake);
%     new_name = split(keyinfo,'_');new_name = new_name{4,1};
%     inciangle = str2double(new_name);inciangle = cast(inciangle,"uint8");
    inciangle = incident(idx);
    r1 = corrcoef(REAL,FAKE);
    Ereal = 20*log10(REAL);Efake = 20*log10(FAKE);
    db_rmse = sqrt(mean(mean((Ereal-Efake).^2)));

    a = [inciangle-5,inciangle+5];
    a(a<=0)=1; a(a>=90)=90;
    ErealMainloab = 20*log10(REAL(a(1):a(2),175:185));
    EfakeMainloab = 20*log10(FAKE(a(1):a(2),175:185));
    r2 = corrcoef(ErealMainloab,EfakeMainloab);
    db_rmse_Mainloab = sqrt(mean(mean((ErealMainloab-EfakeMainloab).^2)));
    
    ARMSE(1,idx) = db_rmse;ARMSE(2,idx) = db_rmse_Mainloab;
    ACorrelationCoefficient(1,idx) = r1(1,2);ACorrelationCoefficient(2,idx) = r2(1,2);
    
    r = 1;num_theta = 91;num_phi= 361;
    theta = linspace(0,pi/2,num_theta);phi = linspace(0,2*pi,num_phi);
    [theta,phi] = meshgrid(theta,phi);
    x = r*sin(theta).*cos(phi);y = r*sin(theta).*sin(phi);z=r*cos(theta);
    fig = figure(1);
    set(fig,'Color','white','Renderer','openGL');
    set(gcf,'unit','normalized','position',[0.1,0.1,0.8,0.5]);
    tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
    nexttile
    surf(x, y, z, REAL');
    shading interp
    axis equal;  % 保持坐标轴比例一致，使球面不会被压缩
    title('真实远场 Efar');axis tight; 
    set(gca, 'XTick', []);  % 去掉X轴刻度
    set(gca, 'YTick', []);  % 去掉X轴刻度
    set(gca, 'ZTick', []);  % 去掉X轴刻度
    xlabel('X');ylabel('Y');zlabel('Z');

    nexttile
    surf(x, y, z, FAKE');
    shading interp
    axis equal;  % 保持坐标轴比例一致，使球面不会被压缩
    title('重建远场 Efar');axis tight; 
    set(gca, 'XTick', []);  % 去掉X轴刻度
    set(gca, 'YTick', []);  % 去掉X轴刻度
    set(gca, 'ZTick', []);  % 去掉X轴刻度
    xlabel('X');ylabel('Y');zlabel('Z');

    colormap('jet');
    c1 = colorbar('Position', [0.93, 0.25, 0.015, 0.5]);  % [x, y, width, height]
    c2 = colorbar('Position', [0.5, 0.25, 0.015, 0.5]);  % [x, y, width, height]
    % c1.Title.String = "A Title";
    set(get(c1,'Title'),'string','E_{far} [V/m]','FontSize',10);
    set(get(c2,'Title'),'string','E_{far} [V/m]','FontSize',10);
    
    drawnow;
    pause(1);
%     saveas(gcf,[fileparts(materialPath),'\allpic\',scenario,'_',num2str(scale),'_',num2str(index) '.jpg']);
    currentFrame = getframe(fig);
    writeVideo(vidObj,currentFrame);
    clf(fig,'reset');
end
close(vidObj);

