%% =========================================================================
% Q-FedSecure DR-XAI: Telemedicine Bandwidth & Rural Queue Simulation
% Smart India Hackathon Prototype - System Simulation Module
%
% Models:
% 1. Multi-Priority Patient Arrival Queues at Rural Primary Health Centers (PHCs)
% 2. 2G / 3G / 4G Telemedicine Uplink Bandwidth & Packet Loss Dynamics
% 3. Cloud Centralized Inference vs. Q-FedSecure Edge Quantum-Classical Sync
% =========================================================================

function matlab_sim()
    clear; clc; close all;
    fprintf('===============================================================\n');
    fprintf('  Q-FedSecure DR-XAI: Telemedicine Bandwidth & Queuing Model   \n');
    fprintf('  Simulation of District Rural Healthcare Tele-Ophthalmology    \n');
    fprintf('===============================================================\n\n');

    %% 1. Simulation Parameters
    num_patients = 1000;           % Number of screened patients in campaign
    arrival_rate = 12;             % Patients per hour arriving at PHC (lambda)
    consultation_rate = 4;         % Ophthalmologist tele-consult capacity/hr (mu)
    num_specialists = 3;           % Available district tele-ophthalmologists (c)

    % Network Parameters for Rural Telemedicine Nodes
    networks = struct();
    networks.G2 = struct('name', '2G (EDGE)', 'bandwidth_kbps', 128,  'packet_loss', 0.12, 'latency_ms', 650);
    networks.G3 = struct('name', '3G (HSPA)', 'bandwidth_kbps', 1500, 'packet_loss', 0.04, 'latency_ms', 180);
    networks.G4 = struct('name', '4G (LTE)',  'bandwidth_kbps', 12000,'packet_loss', 0.01, 'latency_ms', 45);

    % Payload Sizes (in Kilobytes)
    raw_fundus_image_kb = 8500;    % 8.5 MB uncompressed high-resolution fundus
    clahe_compressed_kb = 350;     % 350 KB preprocessed thumbnail for specialist review
    qfed_model_weights_kb = 82;    % 82 KB compressed 4-qubit VQC gradient delta

    %% 2. Transmission Latency & Bandwidth Consumption Simulation
    net_types = {'G2', 'G3', 'G4'};
    t_cloud_transfer = zeros(1, 3);
    t_edge_sync = zeros(1, 3);

    fprintf('--- Part 1: Network Latency Analysis (Payload Transfer Time) ---\n');
    for i = 1:3
        net = networks.(net_types{i});
        eff_bw = net.bandwidth_kbps * (1 - net.packet_loss); % Effective throughput
        
        % Centralized Cloud: Transmit full raw image for inference
        t_cloud = (raw_fundus_image_kb * 8) / eff_bw + (net.latency_ms / 1000);
        t_cloud_transfer(i) = t_cloud;

        % Q-FedSecure Edge: Local inference + periodic 82 KB weight sync
        t_edge = (qfed_model_weights_kb * 8) / eff_bw + (net.latency_ms / 1000);
        t_edge_sync(i) = t_edge;

        fprintf('Network: %-10s | Cloud Upload: %6.2f sec | Q-FedSecure Sync: %5.3f sec | Bandwidth Reduction: %5.1f%%\n', ...
            net.name, t_cloud, t_edge, (1 - (qfed_model_weights_kb / raw_fundus_image_kb)) * 100);
    end

    %% 3. Priority Queuing Simulation (M/M/c Priority Queue)
    % Priority 1 (High): Grade 3 Severe NPDR & Grade 4 Proliferative DR (~15% cohort)
    % Priority 2 (Low): Grade 0 Normal, Grade 1 Mild, Grade 2 Moderate (~85% cohort)
    
    rng(42); % Reproducibility seed
    inter_arrival_times = exprnd(1 / arrival_rate, 1, num_patients);
    arrival_timestamps = cumsum(inter_arrival_times);

    % Class distribution: 0: 60%, 1: 15%, 2: 10%, 3: 10%, 4: 5%
    dr_grades = randsample([0, 1, 2, 3, 4], num_patients, true, [0.60, 0.15, 0.10, 0.10, 0.05]);
    priorities = ones(1, num_patients);
    priorities(dr_grades >= 3) = 1; % Emergency Priority
    priorities(dr_grades < 3) = 2;  % Routine Review

    service_times = exprnd(1 / consultation_rate, 1, num_patients);

    % Queue processing
    waiting_times_conventional = zeros(1, num_patients); % FIFO without AI Triage
    waiting_times_qfed = zeros(1, num_patients);         % Priority Queue with Edge AI Triage

    % Conventional FIFO simulation
    server_free_times = zeros(1, num_specialists);
    for p = 1:num_patients
        [min_free, s_idx] = min(server_free_times);
        start_time = max(arrival_timestamps(p), min_free);
        waiting_times_conventional(p) = (start_time - arrival_timestamps(p)) * 60; % in minutes
        server_free_times(s_idx) = start_time + service_times(p);
    end

    % Q-FedSecure AI-Triaged Priority Queue (Emergency cases jump queue)
    server_free_times_prio = zeros(1, num_specialists);
    % Sort by arrival and priority
    p_emergency = find(priorities == 1);
    p_routine = find(priorities == 2);

    for idx = 1:length(p_emergency)
        p = p_emergency(idx);
        [min_free, s_idx] = min(server_free_times_prio);
        start_time = max(arrival_timestamps(p), min_free);
        waiting_times_qfed(p) = (start_time - arrival_timestamps(p)) * 60;
        server_free_times_prio(s_idx) = start_time + service_times(p);
    end
    for idx = 1:length(p_routine)
        p = p_routine(idx);
        [min_free, s_idx] = min(server_free_times_prio);
        start_time = max(arrival_timestamps(p), min_free);
        waiting_times_qfed(p) = (start_time - arrival_timestamps(p)) * 60;
        server_free_times_prio(s_idx) = start_time + service_times(p);
    end

    fprintf('\n--- Part 2: Triage & Clinical Waiting Time Analysis ---\n');
    fprintf('Conventional FIFO Mean Wait Time:        %6.2f min\n', mean(waiting_times_conventional));
    fprintf('Q-FedSecure AI Critical Patient Wait Time: %6.2f min (Emergency PDR Cases)\n', mean(waiting_times_qfed(priorities == 1)));
    fprintf('Q-FedSecure AI Routine Patient Wait Time:  %6.2f min\n', mean(waiting_times_qfed(priorities == 2)));

    %% 4. Plotting Diagnostic Figures
    figure('Name', 'Q-FedSecure Telemedicine & Queuing Model', 'Position', [100, 100, 1100, 500]);

    % Subplot 1: Network Transmission Time Comparison
    subplot(1, 2, 1);
    b = bar([t_cloud_transfer; t_edge_sync]', 'grouped');
    set(gca, 'XTickLabel', {'2G (EDGE)', '3G (HSPA)', '4G (LTE)'});
    ylabel('Transmission Time per Study (Seconds)', 'FontSize', 11);
    title('Bandwidth Latency: Central Cloud vs Q-FedSecure Edge', 'FontSize', 12, 'FontWeight', 'bold');
    legend({'Central Cloud (8.5 MB Raw Image)', 'Q-FedSecure Edge (82 KB Quantum Weights)'}, 'Location', 'northwest');
    grid on;
    b(1).FaceColor = [0.85, 0.33, 0.10];
    b(2).FaceColor = [0.00, 0.60, 0.50];

    % Subplot 2: Patient Triage Waiting Time Distribution
    subplot(1, 2, 2);
    box_data = [waiting_times_conventional', waiting_times_qfed(priorities == 1)', waiting_times_qfed(priorities == 2)'];
    % Pad with NaNs for unequal lengths
    max_len = max([length(waiting_times_conventional), sum(priorities == 1), sum(priorities == 2)]);
    c1 = [waiting_times_conventional, nan(1, max_len - length(waiting_times_conventional))]';
    c2 = [waiting_times_qfed(priorities == 1), nan(1, max_len - sum(priorities == 1))]';
    c3 = [waiting_times_qfed(priorities == 2), nan(1, max_len - sum(priorities == 2))]';
    
    boxplot([c1, c2, c3], 'Labels', {'Conventional FIFO', 'Q-Fed (Critical PDR)', 'Q-Fed (Routine)'});
    ylabel('Waiting Time to Doctor Review (Minutes)', 'FontSize', 11);
    title('Rural Telemedicine Waiting Time with AI Prioritization', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;

    fprintf('\n[Simulation Complete] Figures plotted successfully.\n');
end
