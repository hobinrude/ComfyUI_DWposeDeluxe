import { app } from "../../scripts/app.js";

const COLOR_THEMES = {
    blue: { nodeColor: "#134679", nodeBgColor: "#003264" },
};

const NODE_COLORS = {
    "DWposeDeluxeNode": "blue",
    "CustomOptions": "blue",
    "KeypointConverter": "blue",
    "LoadPoseKeypoints": "blue",
    "FrameNumberNode": "blue",
    "KeypointPrinter": "blue",
    "PoseInterpolation": "blue"
};

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

let colorKeys = Object.keys(COLOR_THEMES).filter(key => key !== "none");
shuffleArray(colorKeys);

function setNodeColors(node, theme) {
    if (!theme) { return; }
    node.shape = "default";
    if (theme.nodeColor && theme.nodeBgColor) {
        node.color = theme.nodeColor;
        node.bgcolor = theme.nodeBgColor;
    }
}

const ext = {
    name: "DWposeDeluxe.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];

            if (colorKey === "random") {
                if (colorKeys.length === 0 || !COLOR_THEMES[colorKeys[colorKeys.length - 1]]) {
                    colorKeys = Object.keys(COLOR_THEMES).filter(key => key !== "none");
                    shuffleArray(colorKeys);
                }
                colorKey = colorKeys.pop();
            }

            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
            node.size[0] = 280;
        }
    }
};

app.registerExtension(ext);
