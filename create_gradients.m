clear; clc

%%

labeling = load_parcellation('schaefer', 200).schaefer_200;
[surf_lh, surf_rh] = load_conte69();

%%

report_MS = readtable('/home/ws/Documenti/MS_Connectomics/preprocessed_data/ms_NAPOLI_rpt_2023-01-25-121134.xlsx');
report_HC = readtable('/home/ws/Documenti/MS_Connectomics/preprocessed_data/hc_NAPOLI_rpt_2023-01-25-121218.xlsx');

path_MS = dir('/home/ws/Documenti/MS_Connectomics/preprocessed_data/ms/sub*.mat/200cortical_z.mat');
path_HC = dir('/home/ws/Documenti/MS_Connectomics/preprocessed_data/hc/sub*.mat/200cortical_z.mat');

for sub = 1:length(path_MS)
    mat = load([path_MS(sub).folder filesep path_MS(sub).name]).zconmat;
    %mat(1:200+1:200*200) = 1; 
    matconn_MS(sub, :, :) = mat;
    %matconn_MS(sub, :, :) = atanh(matconn_MS(sub, :, :));
end

validSub = true(length(path_MS), 1);
for sub = 1:length(path_MS)
    if any(isnan(matconn_MS(sub, :, :)), 'all')
        validSub(sub) = false;
    end
end
matconn_MS = matconn_MS(validSub, :, :);
report_MS = report_MS(validSub, :);
    
for sub = 1:length(path_HC)
    mat = load([path_HC(sub).folder filesep path_HC(sub).name]).zconmat;
    %mat(1:200+1:200*200) = 1; 
    matconn_HC(sub, :, :) = mat;
    %matconn_HC(sub, :, :) = atanh(matconn_HC(sub, :, :));
end

validSub = true(length(path_HC), 1);
for sub = 1:length(path_HC)
    if any(isnan(matconn_HC(sub, :, :)), 'all')
        validSub(sub) = false;
    end
end
matconn_HC = matconn_HC(validSub, :, :);
report_HC = report_HC(validSub, :);

%%

matconn_avg = squeeze(mean(cat(1, matconn_MS, matconn_HC), 1));

gm = GradientMaps('kernel', 'cosine', 'approach', 'dm', 'random_state', 42);
gm = gm.fit(matconn_avg);

scree_plot(gm.lambda{1});
fig = plot_hemispheres(gm.gradients{1}(:, 1:2), {surf_lh, surf_rh}, 'parcellation', labeling, 'labeltext', {'Gradient 1', 'Gradient 2'});
gradient_in_euclidean(gm.gradients{1}(:, 1:2), {surf_lh, surf_rh}, labeling);


%%

mat = cat(1, matconn_MS, matconn_HC);
matconn = {};
for sub = 1:size(mat, 1)
    matconn = [matconn squeeze(mat(sub, :, :))];
end

align_ind = GradientMaps('kernel', 'cosine', 'approach', 'dm', 'alignment', 'pa', 'random_state', 42);
align_ind = align_ind.fit(matconn, 'reference', gm.gradients{1});

group_gradients = gm.gradients{1};

save('gradients_matlab_200z.mat', align_ind)
save('group_gradients_matlab_200z.mat', group_gradients)