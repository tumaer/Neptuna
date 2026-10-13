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
        video: "https://huggingface.co/datasets/FluidVerse/2D_SRBA_OOOO/resolve/main/SRBA_OOOO_Mas2.10.mp4",
        scores: {
            "ConvNeXt":     { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "CNO":          { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO1":        { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO2":        { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "ScOT":         { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "DPOT-M":       { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "Poseidon-B":   { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]}
        }
    },
    "2D_SABW_SSOO": {
        title: "2D Shock-Induced Air Bubble Collapse in Water with Symmetric BC",
        desc: "The dataset captures the time-evolving behavior of 2D cylindrical air bubbles subjected to an external shock wave in water. The interaction with the shock wave results in a collapse of the air bubble. Here we investigate a scenario with symmetric boundary conditions at the north and south walls.",
        resolution: ["256x256", "512x512"],
        trajectories: "300 Trajectories",
        physics: "Ma = 1.3 - 2.1",
        dimensions: "2D",
        video: "https://huggingface.co/datasets/FluidVerse/2D_SABW_SSOO/resolve/main/2D_SABW_SSOO_Mach2.10.mp4",
        scores: {
            "ConvNeXt":     { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "CNO":          { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO1":        { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO2":        { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "ScOT":         { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "DPOT-M":       { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "Poseidon-B":   { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]}
        }
    },
    "2D_SABW_OOOO": {
        title: "2D Shock-Induced Air Bubble Collapse in Water with Open BC",
        desc: "The dataset captures the time-evolving behavior of 2D cylindrical air bubbles subjected to an external shock wave in water. The interaction with the shock wave results in a collapse of the air bubble. Here we investigate a scenario with open boundary conditions at every wall.",
        resolution: ["256x256", "512x512"],
        trajectories: "300 Trajectories",
        physics: "Ma = 1.3 - 2.1",
        dimensions: "2D",
        video: "https://huggingface.co/datasets/FluidVerse/2D_SABW_OOOO/resolve/main/SABW_OOOO_Mas2.10.mp4",
        scores: {
            "ConvNeXt":     { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "CNO":          { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO1":        { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO2":        { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "ScOT":         { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "DPOT-M":       { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "Poseidon-B":   { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]}
        }
    },
    "2D_SDBA_SSOO": {
        title: "2D Shock-Induced Droplet Breakup in Air with Symmetric BC",
        desc: "The dataset captures the time-evolving behavior of 2D cylindrical droplets subjected to an external shock wave in air. The interaction with the shock wave results in two different breakup modes namely SIE and RTP depending on the weber number. Here we investigate a scenario with symmetric boundary conditions at the north and south walls.",
        resolution: ["256x256", "512x512"],
        trajectories: "300 Trajectories",
        physics: "Ma = 1.2 - 3.2 & We = 10 - 30 / 10k - 30k",
        dimensions: "2D",
        video: "https://huggingface.co/datasets/FluidVerse/2D_SDBA_SSOO/resolve/main/SDBA_SSOO_SIE_Mas1.20.mp4",
        scores: {
            "ConvNeXt":     { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "CNO":          { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO1":        { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO2":        { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
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
        video: "https://huggingface.co/datasets/FluidVerse/3D_SDBA_SSOOSS/resolve/main/SDBA_SSOOSS_SIE_Mas1.20.mp4",
        scores: {
            "ConvNeXt3D":     { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO3D":          { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]}
        }
    },
    "3D_SABW_SSOOSS": {
        title: "3D Shock-Induced Air Bubble Collapse in Water with Symmetric BC",
        desc: "The dataset captures the time-evolving behavior of 3D spherical air bubbles subjected to an external shock wave in water. The interaction with the shock wave results in a collapse of the air bubble. Here we investigate a scenario with symmetric boundary conditions at the north, south, top and bottom walls.",
        resolution: ["128x128x128"],
        trajectories: "180 Trajectories",
        dimensions: "3D",
        physics: "Ma = 1.3 - 2.1",
        video: "https://huggingface.co/datasets/FluidVerse/3D_SABW_SSOOSS/resolve/main/SABW_SSOOSS_Mas1.70.mp4",
        scores: {
            "ConvNeXt3D":     { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]},
            "FFNO3D":          { MSE: [-1, -1, -1, -1, -1], SoftAdapt: [-1, -1, -1, -1, -1], GradNorm: [-1, -1, -1, -1, -1]}
        }
    }
};

// -------------------------------------------------------------
// CONFIG & METADATA
// -------------------------------------------------------------
const models = [
    { id: 'ConvNeXt',   name: 'ConvNeXt',   color: '#00f2fe', visible: true, info: "[ 4, 1, 10 ]", dimensions: "2D" },
    { id: 'ConvNeXt3D',   name: 'ConvNeXt',   color: '#00f2fe', visible: true, info: "[ 4, 1, 5 ]", dimensions: "3D" },
    { id: 'CNO',        name: 'CNO',        color: '#9d4edd', visible: true, info: "[ 4, 1, 10 ]", dimensions: "2D" },
    { id: 'FFNO1',      name: 'FFNO',       color: '#10b981', visible: false, info: "[ 4, 1, 10 ]", dimensions: "2D" },
    { id: 'FFNO2',      name: 'FFNO',       color: '#f59e0b', visible: false, info: "[ 10, 1, 4 ]", dimensions: "2D" },
    { id: 'FFNO3D',      name: 'FFNO',       color: '#f59e0b', visible: false, info: "[ 4, 1, 5 ]", dimensions: "3D" },
    { id: 'ScOT',       name: 'ScOT',       color: '#3b82f6', visible: false, info: "[ 4, 1, 10 ]", dimensions: "2D" },
    { id: 'DPOT-M',     name: 'DPOT-M (Finetuned)',     color: '#f43f5e', visible: false, info: "[ 10, 1, 4 ]", dimensions: "2D" },
    { id: 'Poseidon-B', name: 'Poseidon-B (Finetuned)', color: '#ca62b2', visible: false, info: "[ 1, 1, 10 ]", dimensions: "2D" }
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