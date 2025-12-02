import { app } from "../../scripts/app.js";

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true);
}

const linkWidgetVisibility = (node, parentWidget, childWidget, condition) => {
    if (!parentWidget || !childWidget) return;

    if (childWidget.origType === undefined) {
        childWidget.origType = childWidget.type;
    }
    if (childWidget.origComputeSize === undefined) {
        childWidget.origComputeSize = childWidget.computeSize;
    }

    const onParentChange = (value) => {
        const shouldBeVisible = condition ? condition(value) : value === true;

        if (shouldBeVisible) {
            childWidget.type = childWidget.origType;
            childWidget.computeSize = childWidget.origComputeSize;
        } else {

            if (childWidget.origType === "BOOLEAN" || childWidget.origType === "CHECKBOX") {
                childWidget.value = false;
            }

            childWidget.type = "hidden";
            childWidget.computeSize = () => [0, -4];
        }
        fitHeight(node);
    };

    const origCallback = parentWidget.callback;
    parentWidget.callback = (value) => {
        origCallback?.call(node, value);
        onParentChange(value);
    };

    setTimeout(() => {
        onParentChange(parentWidget.value);
    }, 1);
};

app.registerExtension({
    name: "dwpose_adv_ui",
    async nodeCreated(node) {
        if (node.comfyClass === "DWposeDeluxeNode") {
            const providerWidget = node.widgets.find(w => w.name === "provider_type");
            const precisionWidget = node.widgets.find(w => w.name === "precision");

            const showBodyWidget = node.widgets.find(w => w.name === "show_body");
            const showFeetWidget = node.widgets.find(w => w.name === "show_feet");
            const showFaceWidget = node.widgets.find(w => w.name === "show_face");
            const showHandsWidget = node.widgets.find(w => w.name === "show_hands");

            const bodyThresholdWidget = node.widgets.find(w => w.name === "body_threshold");
            const faceThresholdWidget = node.widgets.find(w => w.name === "face_threshold");
            const handThresholdWidget = node.widgets.find(w => w.name === "hand_threshold");

            const updateModelLists = async () => {
                const detectorModelWidget = node.widgets.find(w => w.name === "detector_model");
                const estimatorModelWidget = node.widgets.find(w => w.name === "estimator_model");

                const providerType = providerWidget.value;
                const precision = precisionWidget.value;

                let url = `/dwpose_adv/get_model_list?provider_type=${providerType}`;
                if (providerType === "GPU" && precision) {
                    url += `&precision=${precision}`;
                }

                try {
                    const response = await fetch(url);
                    const data = await response.json();
                    
                    if (detectorModelWidget) {
                        const currentDetectorValue = detectorModelWidget.value;
                        detectorModelWidget.options.values = data.detector_models;
                        if (!data.detector_models.includes(currentDetectorValue)) {
                            detectorModelWidget.value = detectorModelWidget.options.values[0];
                        } else {
                            detectorModelWidget.value = currentDetectorValue;
                        }
                    }

                    if (estimatorModelWidget) {
                        const currentEstimatorValue = estimatorModelWidget.value;
                        estimatorModelWidget.options.values = data.estimator_models;
                        if (!data.estimator_models.includes(currentEstimatorValue)) {
                            estimatorModelWidget.value = estimatorModelWidget.options.values[0];
                        } else {
                            estimatorModelWidget.value = currentEstimatorValue;
                        }
                    }
                } catch (error) {
                    console.error("Error fetching model lists:", error);
                    if (detectorModelWidget) detectorModelWidget.options.values = ["Error"];
                    if (estimatorModelWidget) estimatorModelWidget.options.values = ["Error"];
                }
                fitHeight(node);
            };

            const origProviderCallback = providerWidget.callback;
            providerWidget.callback = (value) => {
                origProviderCallback?.call(node, value);
                updateModelLists();
            };

            const origPrecisionCallback = precisionWidget.callback;
            precisionWidget.callback = (value) => {
                origPrecisionCallback?.call(node, value);
                updateModelLists();
            };
            
            linkWidgetVisibility(node, providerWidget, precisionWidget, (val) => val === "GPU");
            linkWidgetVisibility(node, showBodyWidget, showFeetWidget);
            linkWidgetVisibility(node, showBodyWidget, bodyThresholdWidget);
            linkWidgetVisibility(node, showFaceWidget, faceThresholdWidget);
            linkWidgetVisibility(node, showHandsWidget, handThresholdWidget);
            
            setTimeout(() => {
                updateModelLists();
            }, 1);

            node.onExecuted = async function(message) {
                let refreshNeeded = false;
                if (Array.isArray(message?.ui)) {
                    for (const ui_item of message.ui) {
                        if (ui_item === "model_refresh_needed") {
                            refreshNeeded = true;
                            break;
                        }
                    }
                }

                if (refreshNeeded) {
                    const originalProviderValue = providerWidget.value;
                    providerWidget.value = (originalProviderValue === "CPU") ? "GPU" : "CPU";
                    providerWidget.callback(providerWidget.value);
                    providerWidget.value = originalProviderValue;
                    providerWidget.callback(originalProviderValue);
                    await updateModelLists();
                }
            };
        }
    }
});