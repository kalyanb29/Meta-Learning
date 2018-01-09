load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori_ws',filesep,'Obj-5',filesep,'run-1',filesep,'Params.mat'));
archivef = [];
prob = load_problem_definition(def);
for i = 1:25
    archive = [];
    load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori_ws',filesep,'Obj-5',filesep,'run-',num2str(i),filesep,'Archive.mat'))
    archive1 = archive(1:2000,:);
    archive1feas = archive1(archive1(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
    if ~isempty(archive1feas)
        [front,~] = nd_sort(archive1feas,(1:size(archive1feas,1))');
        archivef = [archivef;archive1feas(front(1).f,:)];
    end
end
[front,~] = nd_sort(archivef,(1:size(archivef,1))');
archivend = archivef(front(1).f,:);
save('archive_wsnd.mat','archivend')

load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori',filesep,'Obj-5',filesep,'run-1',filesep,'Params.mat'));
archivef = [];
prob = load_problem_definition(def);
for i = 1:25
    archive = [];
    load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori',filesep,'Obj-5',filesep,'run-',num2str(i),filesep,'Archive.mat'))
    archive1 = archive(1:2000,:);
    archive1feas = archive1(archive1(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
    if ~isempty(archive1feas)
        [front,~] = nd_sort(archive1feas,(1:size(archive1feas,1))');
        archivef = [archivef;archive1feas(front(1).f,:)];
    end
end
[front,~] = nd_sort(archivef,(1:size(archivef,1))');
archivend1 = archivef(front(1).f,:);
save('archive_nd.mat','archivend1')
archivetot = [archivend;archivend1];
[front,~] = nd_sort(archivetot,(1:size(archivetot,1))');
archivendtot = archivetot(front(1).f,:);
ideal = min(archivendtot,[],1)
nadir = max(archivendtot,[],1)

pathtest = strcat(pwd,filesep,'Methods',filesep,'Hypervolume',filesep,'test.exe');
str = '%21.20f ';
for j = 1:size(ideal,2)-1
    str = strcat(str,' %21.20f');
end
refstr = '"1.1"';
for ii = 1:size(ideal,2)-1
    refstr = strcat(refstr,' "1.1"');
end
HVws = [];HV = [];
for i = 1:25
    archive = [];
    load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori_ws',filesep,'Obj-5',filesep,'run-',num2str(i),filesep,'Archive.mat'))
    archive1 = archive(1:2000,:);
    archive1feas = archive1(archive1(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
    if ~isempty(archive1feas)
        [front,~] = nd_sort(archive1feas,(1:size(archive1feas,1))');
        archivef = archive1feas(front(1).f,:);
		if ~isempty(archivef)
			f1a = set_extract(archivef,1.1*nadir,0);
			if isempty(f1a)
				f1s = value2quantile(f1a,1,[ideal;nadir]);
				commandhvarchive = strcat('"',pathtest,'"',' "fndarchive1',num2str(i),'.txt"',{' '},refstr);
				fp = fopen(strcat('fndarchive1',num2str(i),'.txt'),'w');
				fprintf(fp,'%s\r\n','#');
				fprintf(fp,strcat(str,'\r\n'),f1s');
				fprintf(fp,'%s','#');
				fclose(fp);
				[~,hv1char] = system(commandhvarchive{1});
				hv1 = str2double(hv1char);
				HVws = [HVws;hv1];
			else
				HVws = [HVws;0];
			end
		else
			HVws = [HVws;0];
		end
    end
end

for i = 1:25
    archive = [];
    load(strcat(pwd,filesep,'Results',filesep,'vehicle_architecture_ori',filesep,'Obj-5',filesep,'run-',num2str(i),filesep,'Archive.mat'))
    archive1 = archive(1:2000,:);
    archive1feas = archive1(archive1(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
    if ~isempty(archive1feas)
        [front,~] = nd_sort(archive1feas,(1:size(archive1feas,1))');
        archivef = archive1feas(front(1).f,:);
		if ~isempty(archivef)
			f1a = set_extract(archivef,1.1*nadir,0);
			if isempty(f1a)
				f1s = value2quantile(f1a,1,[ideal;nadir]);
				commandhvarchive = strcat('"',pathtest,'"',' "fndarchive1',num2str(i),'.txt"',{' '},refstr);
				fp = fopen(strcat('fndarchive1',num2str(i),'.txt'),'w');
				fprintf(fp,'%s\r\n','#');
				fprintf(fp,strcat(str,'\r\n'),f1s');
				fprintf(fp,'%s','#');
				fclose(fp);
				[~,hvchar] = system(commandhvarchive{1});
				hv = str2double(hvchar);
				HV = [HV;hv];
			else
				HV = [HV;0];
			end
		else
			HV = [HV;0];
		end
    end
end

HVwsstat = [max(HVws) mean(HVws) median(HVws) min(HVws) std(HVws)];
HVstat = [max(HV) mean(HV) median(HV) min(HV) std(HV)];
[~,idsortws] = sort(HVws,'descend')
[~,idsort] = sort(HV,'descend')
med_runws = idsortws(13)
med_run = idsort(13)
save('HVws.mat','HVws')
save('HV.mat','HV')
save('HVwsstat.mat','HVwsstat')
save('HVstat.mat','HVstat')