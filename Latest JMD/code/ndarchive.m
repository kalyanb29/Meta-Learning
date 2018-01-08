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
archivend = archivef(front(1).f,:);
save('archive_wsnd.mat','archivend')