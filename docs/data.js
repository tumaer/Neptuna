// -------------------------------------------------------------
// DATASETS DATABASE
// -------------------------------------------------------------
const datasetsData = {
    "2D_SRBA_OOOO": {
        title: "2D Shock-Induced R22 Bubble Collapse in Air with Open BC",
        desc: "The dataset captures the time-evolving behavior of 2D cylindrical R22 bubbles subjected to an external shock wave in air. The interaction with the shock wave results in a collapse of the R22 bubble. Here we investigate a scenario with open boundary conditions at every wall.",
        resolution: ["256x256", "512x512"],
        trajectories: "300 Trajectories",
        physics: "Ma = 1.3 - 2.1",
        dimensions: "2D",
        video: "assets/videos/SRBA_OOOO_Mas2.10.mp4",
        scores: {
            "ConvNeXt":     { MSE: [0.5873, 0.9971, 0.9895, 0.7813, 0.739], SoftAdapt: [0.8344, 0.9977, 0.9837, 0.9271, 0.7647], GradNorm: [0.7666, 0.9976, 0.9756, 0.9055, 0.7678]},
            "CNO":          { MSE: [0.7652, 0.9948, 0.9826, 0.6474, 0.6756], SoftAdapt: [0.9853, 0.9966, 0.9829, 0.8661, 0.7439], GradNorm: [0.9168, 0.9958, 0.9762, 0.8118, 0.7261]},
            "FFNO1":        { MSE: [0.727, 0.9984, 0.9898, 0.8934, 0.7849], SoftAdapt: [1.0, 0.9988, 0.9975, 0.9551, 0.8048], GradNorm: [0.9386, 0.9987, 0.9955, 0.9459, 0.8028]},
            "FFNO2":        { MSE: [0.4467, 0.997, 0.9804, 0.6965, 0.7343], SoftAdapt: [0.6146, 0.9969, 0.9735, 0.7931, 0.7582], GradNorm: [0.7242, 0.9987, 0.9901, 0.8869, 0.8235]},
            "ScOT":         { MSE: [0.6599, 0.9987, 0.9931, 0.9014, 0.7979], SoftAdapt: [0.7766, 0.9973, 0.9765, 0.8967, 0.7574], GradNorm: [0.7249, 0.997, 0.9624, 0.8683, 0.7569]},
            "DPOT-M":       { MSE: [0.6669, 0.9995, 0.994, 0.9151, 0.9967], SoftAdapt: [0.8483, 1.0, 0.998, 1.0, 1.0], GradNorm: [0.7655, 0.9995, 1.0, 0.9692, 0.8574]},
            "Poseidon-B":   { MSE: [0.0, 0.0, 0.0, 0.0, 0.6211], SoftAdapt: [0.1874, 0.0576, 0.082, 0.0013, 0.6256], GradNorm: [0.1311, 0.1112, 0.0352, 0.0112, 0.0]}
        }
    },
    "2D_SABW_SSOO": {
        title: "2D Shock-Induced Air Bubble Collapse in Water with Symmetric BC",
        desc: "The dataset captures the time-evolving behavior of 2D cylindrical air bubbles subjected to an external shock wave in water. The interaction with the shock wave results in a collapse of the air bubble. Here we investigate a scenario with symmetric boundary conditions at the north and south walls.",
        resolution: ["256x256", "512x512"],
        trajectories: "300 Trajectories",
        physics: "Ma = 1.3 - 2.1",
        dimensions: "2D",
        video: "assets/videos/2D_SABW_SSOO_Mach2.10.mp4",
        scores: {
            "ConvNeXt":     { MSE: [0.5825, 0.9953, 0.9506, 0.8192, 0.7688], SoftAdapt: [0.7871, 0.997, 0.9428, 0.9442, 0.8052], GradNorm: [0.7078, 0.996, 0.9377, 0.8852, 0.8506]},
            "CNO":          { MSE: [0.8022, 0.993, 0.9429, 0.6316, 0.7827], SoftAdapt: [0.9988, 0.9965, 0.9585, 0.8181, 0.7664], GradNorm: [1.0, 0.9972, 0.9634, 0.8408, 0.8925]},
            "FFNO1":        { MSE: [0.7157, 0.9992, 0.9775, 0.9545, 0.8709], SoftAdapt: [0.9684, 1.0, 0.9823, 0.9842, 0.9062], GradNorm: [0.9411, 0.9987, 0.9882, 0.9198, 0.8976]},
            "FFNO2":        { MSE: [0.5454, 0.9989, 0.9665, 0.7875, 0.8721], SoftAdapt: [0.3734, 0.9855, 0.8001, 0.0, 0.3787], GradNorm: [0.6982, 0.9988, 0.971, 0.8079, 0.9441]},
            "ScOT":         { MSE: [0.6678, 0.9965, 0.9666, 0.9331, 0.8626], SoftAdapt: [0.8189, 0.9975, 0.9425, 0.988, 0.8437], GradNorm: [0.7196, 0.996, 0.9043, 0.8705, 0.8699]},
            "DPOT-M":       { MSE: [0.6483, 0.9992, 1.0, 0.9321, 1.0], SoftAdapt: [0.7758, 0.9992, 0.98, 1.0, 0.9785], GradNorm: [0.7238, 0.9989, 0.9874, 0.9419, 0.9803]},
            "Poseidon-B":   { MSE: [0.0, 0.0, 0.1139, 0.0866, 0.0839], SoftAdapt: [0.1893, 0.0166, 0.0, 0.0587, 0.0109], GradNorm: [0.1539, 0.1234, 0.041, 0.0528, 0.0]}
        }
    },
    "2D_SABW_OOOO": {
        title: "2D Shock-Induced Air Bubble Collapse in Water with Open BC",
        desc: "The dataset captures the time-evolving behavior of 2D cylindrical air bubbles subjected to an external shock wave in water. The interaction with the shock wave results in a collapse of the air bubble. Here we investigate a scenario with open boundary conditions at every wall.",
        resolution: ["256x256", "512x512"],
        trajectories: "300 Trajectories",
        physics: "Ma = 1.3 - 2.1",
        dimensions: "2D",
        video: "assets/videos/SABW_OOOO_Mas2.10.mp4",
        scores: {
            "ConvNeXt":     { MSE: [0.5914, 0.9976, 0.939, 0.9233, 0.7729], SoftAdapt: [0.7768, 0.9977, 0.9413, 0.9708, 0.7941], GradNorm: [0.7095, 0.9976, 0.9419, 0.9486, 0.8077]},
            "CNO":          { MSE: [0.8218, 0.9956, 0.9467, 0.8515, 0.7557], SoftAdapt: [0.9885, 0.9961, 0.9356, 0.9104, 0.7434], GradNorm: [1.0, 0.9974, 0.955, 0.9355, 0.8455]},
            "FFNO1":        { MSE: [0.7416, 0.9989, 0.9737, 0.9776, 0.8503], SoftAdapt: [0.9882, 0.999, 0.9804, 0.9939, 0.8762], GradNorm: [0.9675, 0.9987, 0.9834, 0.9737, 0.8612]},
            "FFNO2":        { MSE: [0.0, 0.9679, 0.3744, 0.0, 0.0], SoftAdapt: [0.1345, 0.9792, 0.4183, 0.0602, 0.1308], GradNorm: [0.4677, 0.996, 0.8761, 0.5713, 0.7825]},
            "ScOT":         { MSE: [0.6355, 0.9983, 0.9593, 0.9464, 0.8464], SoftAdapt: [0.8118, 0.9972, 0.9195, 0.9529, 0.7843], GradNorm: [0.6892, 0.9975, 0.903, 0.9349, 0.8459]},
            "DPOT-M":       { MSE: [0.6684, 1.0, 1.0, 1.0, 1.0], SoftAdapt: [0.7997, 0.9991, 0.9818, 0.9998, 0.9105], GradNorm: [0.715, 0.999, 0.9873, 0.9739, 0.9458]},
            "Poseidon-B":   { MSE: [0.0017, 0.0, 0.0, 0.5721, 0.153], SoftAdapt: [0.2558, 0.4833, 0.2217, 0.6469, 0.2097], GradNorm: [0.1716, 0.2285, 0.1617, 0.6176, 0.175]}
        }
    },
    "2D_SDBA_SSOO": {
        title: "2D Shock-Induced Droplet Breakup in Air with Symmetric BC",
        desc: "The dataset captures the time-evolving behavior of 2D cylindrical droplets subjected to an external shock wave in air. The interaction with the shock wave results in two different breakup modes namely SIE and RTP depending on the weber number. Here we investigate a scenario with symmetric boundary conditions at the north and south walls.",
        resolution: ["256x256", "512x512"],
        trajectories: "300 Trajectories",
        physics: "Ma = 1.2 - 3.2 & We = 10 - 30 / 10k - 30k",
        dimensions: "2D",
        video: "assets/videos/SDBA_SSOO_SIE_Mas1.20.mp4",
        scores: {
            "ConvNeXt":     { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "CNO":          { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO1":        { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO2":        { MSE: [0.0, 0.0, 0.0, 0.0, 0.0], SoftAdapt: [0.2638, 0.6738, 0.1814, 0.1726, 0.3726], GradNorm: [1.0, 1.0, 1.0, 1.0, 1.0]},
            "ScOT":         { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "DPOT-M":       { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "Poseidon-B":   { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]}
        }
    },
    "3D_SDBA_SSOOSS": {
        title: "3D Shock-Induced Droplet Breakup in Air with Symmetric BC",
        desc: "The dataset captures the time-evolving behavior of 3D spherical droplets subjected to an external shock wave in air. When a shock wave impacts a droplet, the initial response—largely independent of the Weber number—is a deformation phase in which the droplet flattens. This interaction with the shock wave results in two different breakup-modes of the droplet (SIE and RTP). Here we investigate a scenario with symmetric boundary conditions at the north, south, top and bottom walls.",
        resolution: ["128x128x128"],
        trajectories: "180 Trajectories",
        physics: "Ma = 1.2 - 3.2 & We = 10 - 30 / 10k - 30k",
        dimensions: "3D",
        video: "assets/videos/SDBA_SSOOSS_SIE_Mas1.20.mp4",
        scores: {
            "ConvNeXt3D":     { MSE: [0.0, 1.0, 0.2281, 0.0, 1.0], SoftAdapt: [0.8611, 1.0, 0.2849, 0.9829, 1.0], GradNorm: [0.5566, 0.0, 0.0, 0.3694, 0.0]},
            "FFNO3D":          { MSE: [0.7085, 1.0, 1.0, 0.9123, 1.0], SoftAdapt: [0.9997, 1.0, 0.9453, 1.0, 1.0], GradNorm: [1.0, 1.0, 0.9436, 0.9103, 1.0]}
        }
    },
    "3D_SABW_SSOOSS": {
        title: "3D Shock-Induced Air Bubble Collapse in Water with Symmetric BC",
        desc: "The dataset captures the time-evolving behavior of 3D spherical air bubbles subjected to an external shock wave in water. The interaction with the shock wave results in a collapse of the air bubble. Here we investigate a scenario with symmetric boundary conditions at the north, south, top and bottom walls.",
        resolution: ["128x128x128"],
        trajectories: "180 Trajectories",
        dimensions: "3D",
        physics: "Ma = 1.3 - 2.1",
        video: "assets/videos/SABW_SSOOSS_Mas1.70.mp4",
        scores: {
            "ConvNeXt3D":     { MSE: [0.0, 0.0, 0.596, 0.0, 0.0], SoftAdapt: [0.2818, 0.1887, 0.0, 0.2494, 0.1905], GradNorm: [0.4724, 0.5467, 0.3244, 0.634, 0.3665]},
            "FFNO3D":          { MSE: [0.5245, 0.8513, 1.0, 0.8486, 0.8866], SoftAdapt: [1.0, 0.9756, 0.9197, 1.0, 0.9732], GradNorm: [0.9382, 1.0, 0.9813, 0.978, 1.0]}
        }
    }
};

// -------------------------------------------------------------
// CONFIG & METADATA
// -------------------------------------------------------------
const models = [
    { id: 'ConvNeXt',   name: 'ConvNeXt',   color: '#00959d', visible: true, info: "4, 1, 10", dimensions: "2D", fineTune: false },
    { id: 'ConvNeXt3D',   name: 'ConvNeXt',   color: '#00959d', visible: true, info: "4, 1, 5", dimensions: "3D", fineTune: false },
    { id: 'CNO',        name: 'CNO',        color: '#9d4edd', visible: true, info: "4, 1, 10", dimensions: "2D", fineTune: false },
    { id: 'FFNO1',      name: 'FFNO',       color: '#10b935', visible: false, info: "4, 1, 10", dimensions: "2D", fineTune: false },
    { id: 'FFNO2',      name: 'FFNO',       color: '#f59e0b', visible: false, info: "10, 1, 4", dimensions: "2D", fineTune: false },
    { id: 'FFNO3D',      name: 'FFNO',       color: '#f59e0b', visible: false, info: "4, 1, 5", dimensions: "3D", fineTune: false },
    { id: 'ScOT',       name: 'ScOT',       color: '#3b82f6', visible: false, info: "4, 1, 10", dimensions: "2D", fineTune: false },
    { id: 'DPOT-M',     name: 'DPOT-M',     color: '#f43f5e', visible: false, info: "10, 1, 4", dimensions: "2D", fineTune: true },
    { id: 'Poseidon-B', name: 'Poseidon-B', color: '#ca62b2', visible: false, info: "1, 1, 10", dimensions: "2D", fineTune: true }
];

// Training Strategies with distinct visual line styles
const strategies = [
    { id: 'MSE',       name: 'MSE',       dash: [],      fillAlpha: 0.12, visible: true },
    { id: 'SoftAdapt', name: 'SoftAdapt', dash: [6, 4],  fillAlpha: 0.05, visible: true },
    { id: 'GradNorm',  name: 'GradNorm',  dash: [2, 3],  fillAlpha: 0.0,  visible: false }
];

const radarMetrics = [
    'MLW',
    'VRMSE',
    'iqRMSE-enstrophy',
    'SSIM',
    'IRMSE',
];
