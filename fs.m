clear;
data_dir = './dataset/';
% Get file name in dataset data_dir
file_list = dir(data_dir);
file_list = file_list(3:end);

exp_size = 1000;
rep_size = 10;
pctRunOnAll warning('off');

for k = 1:length(file_list)
  k
  file_name = file_list(k).name;
  file_path = strcat(data_dir, file_name);
  load(file_path);
  perf_mat = zeros(exp_size ,4);
  perf_std = zeros(exp_size, 4);
  col = size(data, 2);
  lcol = size(answer, 2);
  parfor m = 1:exp_size
    f_sel = randsample(col, 50);
    exp_mat = zeros(rep_size, 4);
    for n = 1:rep_size
      tr_data = data(sim_seq(:, k), :);
      tr_ans = answer(sim_seq(:, k), :);
      ts_data = data(~sim_seq(:, k), :);
      ts_ans = answer(~sim_seq(:, k), :);

      [pre, post] = mlnb(tr_data(:, f_sel), tr_ans, ts_data(:, f_sel));
      exp_mat(n, 1) = mlacc(ts_ans, pre);
      exp_mat(n, 2) = onerr(ts_ans, post);
      exp_mat(n, 3) = rloss(ts_ans, post);
      exp_mat(n, 4) = mlcov(ts_ans, post) / lcol;
    end
    perf_mat(m, :) = mean(exp_mat);
    perf_std(m, :) = std(exp_mat);
  end
  perf_rand(k) = struct('m', 0, 's', 0);
  perf_rand(k).m = perf_mat;
  perf_rand(k).s = perf_std;
end

save('exp_res.mat', 'perf_rand');



