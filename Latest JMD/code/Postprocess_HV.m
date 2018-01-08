load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori_ws',filesep,'Obj-5',filesep,'run-1',filesep,'Params.mat'));
load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori_ws',filesep,'Obj-5',filesep,'run-1',filesep,'Archive.mat'));
archive1 = archive;
load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori',filesep,'Obj-5',filesep,'run-1',filesep,'Archive.mat'));
archive2 = archive;
load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori_n',filesep,'Obj-5',filesep,'run-1',filesep,'Archive.mat'));
archive3 = archive;
prob = load_problem_definition(def);
costall = [0 200 500 1000 2000 3000 4000];
numnd = [];
archivei1 = archive1(1:costall(end),:);
archivei2 = archive2(1:costall(end),:);
archivei3 = archive3(1:costall(end),:);
archiveall = [archivei1;archivei2;archivei3];
fall = archiveall(archiveall(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
xall = archiveall(archiveall(:,end) == 0,2:1+prob.nx);
[xa,idxu] = unique(xall,'rows');
fa = fall(idxu,:);
[front,~] = nd_sort(fa,(1:size(fa,1))');
xa_all = xa(front(1).f,:);
fa_all = fa(front(1).f,:);
ideal = min(fa_all,[],1);
nadir = max(fa_all,[],1);
numhv1 = []; numhv2 = [];numhv3 = [];
pathtest = strcat(pwd,filesep,'Methods',filesep,'Hypervolume',filesep,'test.exe');
for i = 1:numel(costall)
% 1st archive
    str = '%21.20f ';
    for j = 1:size(fa_all,2)-1
        str = strcat(str,' %21.20f');
    end
    refstr = '"1.1"';
    for ii = 1:size(fa_all,2)-1
        refstr = strcat(refstr,' "1.1"');
    end
    archivei11 = archivei1(1:costall(i),:);
    xall = [];fall = [];
    xall = archivei11(archivei11(:,end) == 0,2:1+prob.nx);
    fall = archivei11(archivei11(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
    [xa1,idxu] = unique(xall,'rows');
    fa1 = fall(idxu,:);
    [front,~] = nd_sort(fa1,(1:size(fa1,1))');
    if ~isempty(front)
        f1 = fa1(front(1).f,:);
        f1a = set_extract(f1,1.1*nadir,0);
        f1s = value2quantile(f1a,1,[ideal;nadir]);
        commandhvarchive = strcat('"',pathtest,'"',' "fndarchive1',num2str(i),'.txt"',{' '},refstr);
        fp = fopen(strcat('fndarchive1',num2str(i),'.txt'),'w');
        fprintf(fp,'%s\r\n','#');
        fprintf(fp,strcat(str,'\r\n'),f1s');
        fprintf(fp,'%s','#');
        fclose(fp);
        [~,hv1char] = system(commandhvarchive{1});
        hv1 = str2double(hv1char);
        numhv1 = [numhv1;hv1];
    else
        numhv1 = [numhv1;0];
    end
% 2nd archive
    archivei22 = archivei2(1:costall(i),:);
    xall = [];fall = [];
    xall = archivei22(archivei22(:,end) == 0,2:1+prob.nx);
    fall = archivei22(archivei22(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
    [xa2,idxu] = unique(xall,'rows');
    fa2 = fall(idxu,:);
    [front,~] = nd_sort(fa2,(1:size(fa2,1))');
    if ~isempty(front)
        f2 = fa2(front(1).f,:);
        f2a = set_extract(f2,1.1*nadir,0);
        f2s = value2quantile(f2a,1,[ideal;nadir]);
        commandhvarchive = strcat('"',pathtest,'"',' "fndarchive2',num2str(i),'.txt"',{' '},refstr);
        fp = fopen(strcat('fndarchive2',num2str(i),'.txt'),'w');
        fprintf(fp,'%s\r\n','#');
        fprintf(fp,strcat(str,'\r\n'),f2s');
        fprintf(fp,'%s','#');
        fclose(fp);
        [~,hv2char] = system(commandhvarchive{1});
        hv2 = str2double(hv2char);
        numhv2 = [numhv2;hv2];
    else
        numhv2 = [numhv2;0];
    end
   % 3rd archive
    archivei33 = archivei3(1:costall(i),:);
    xall = [];fall = [];
    xall = archivei33(archivei33(:,end) == 0,2:1+prob.nx);
    fall = archivei33(archivei33(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
    [xa3,idxu] = unique(xall,'rows');
    fa3 = fall(idxu,:);
    [front,~] = nd_sort(fa3,(1:size(fa3,1))');
    if ~isempty(front)
        f3 = fa3(front(1).f,:);
        f3a = set_extract(f3,1.1*nadir,0);
        f3s = value2quantile(f3a,1,[ideal;nadir]);
        commandhvarchive = strcat('"',pathtest,'"',' "fndarchive3',num2str(i),'.txt"',{' '},refstr);
        fp = fopen(strcat('fndarchive3',num2str(i),'.txt'),'w');
        fprintf(fp,'%s\r\n','#');
        fprintf(fp,strcat(str,'\r\n'),f3s');
        fprintf(fp,'%s','#');
        fclose(fp);
        [~,hv3char] = system(commandhvarchive{1});
        hv3 = str2double(hv3char);
        numhv3 = [numhv3;hv3];
    else
        numhv3 = [numhv3;0];
    end
end
plot(costall,numhv2','ro-');hold on;
plot(costall,numhv1','bo-');
plot(costall,numhv3','mo-');
xlabel('Evaluation Cost','fontname','arial','fontsize',16);
ylabel('Hypervolume','fontname','arial','fontsize',16);
legendtxt = {'SaMaO-ASF+ED','MaO','NSGA-II'};
legend(legendtxt,'fontname','arial','fontsize',16);
set(gca,'fontname','arial','fontsize',16);
grid on
saveas(gcf,'vehicle_HV.fig');
saveas(gcf,'vehicle_HV.eps','epsc');