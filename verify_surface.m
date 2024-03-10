% 探究面积和计算值的出入
currentDir = pwd;

% 进入上级目录
parentDir = fileparts(currentDir);
folderPath = [parentDir,'\Simulation','\Azimuth_135'];
fileList = dir(fullfile(folderPath, '**', '*.ol'));

LIST = [];
olFiles={};
for i = 1:numel(fileList)

     olFiles{i} =[fileList(i).folder,'\',fileList(i).name];

   
    
end

for i = 1:length(olFiles)
    fid=fopen(olFiles{i},'r');
    B=textscan(fid,'%f %f %f %f %f %f %f','Headerlines',13);%把ffe文件的数据导入
    fclose(fid);
    clear result;
    for a=1:7
        result(:,a)=B{1,a};
    end
    X = result(:,2)*1000;
    Y = result(:,3)*1000;
    S = result(:,7);
    Smean = mean(S);
    LIST(i) = Smean*size(result,1);
    
end



