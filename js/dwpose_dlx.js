import { app } from "../../scripts/app.js";

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true);
}

app.registerExtension({
    name: "dwpose_adv_ui",
    async nodeCreated(node) {
        if (node.comfyClass !== "DWposeDeluxeNode") return;

        const providerWidget = node.widgets.find(w => w.name === "provider_type");
        const precisionWidget = node.widgets.find(w => w.name === "precision");
        const numberedCompositeWidget = node.widgets.find(w => w.name === "numbered_composite");
        const cornerPositionWidget = node.widgets.find(w => w.name === "corner_position");

        // Store original index and a reference to the original precision widget object
        const precisionWidgetOriginalIndex = node.widgets.findIndex(w => w.name === "precision");
        const precisionWidgetReference = precisionWidget;

        // Store original index and a reference to the original corner_position widget object
        const cornerPositionWidgetOriginalIndex = node.widgets.findIndex(w => w.name === "corner_position");
        const cornerPositionWidgetReference = cornerPositionWidget;

        const updateModelLists = async () => {
            // Address by name INSIDE the function to get the live widgets
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
                                console.log("updateModelLists: API response data", data);
                // Update detector_model options
                if (detectorModelWidget) {
                    const currentDetectorValue = detectorModelWidget.value;
                    detectorModelWidget.options.values = data.detector_models;
                    if (!data.detector_models.includes(currentDetectorValue)) {
                        detectorModelWidget.value = data.detector_models[0];
                    } else {
                        detectorModelWidget.value = currentDetectorValue;
                    }
                }

                // Update estimator_model options
                if (estimatorModelWidget) {
                    const currentEstimatorValue = estimatorModelWidget.value;
                    estimatorModelWidget.options.values = data.estimator_models;
                    if (!data.estimator_models.includes(currentEstimatorValue)) {
                        estimatorModelWidget.value = data.estimator_models[0];
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

            if (value === "GPU") {
                // If the precision widget is currently removed, re-insert it
                if (node.widgets.findIndex(w => w.name === "precision") === -1) {
                    node.widgets.splice(node.widgets.findIndex(w => w.name === "provider_type") + 1, 0, precisionWidgetReference);
                    precisionWidgetReference.value = "fp32"; // Explicitly set value after re-insertion
                    fitHeight(node); // Adjust height immediately after showing
                }
            } else { // Condition to hide precision widget
                // If the precision widget is currently present, remove it
                const currentPrecisionWidgetIndex = node.widgets.findIndex(w => w.name === "precision");
                if (currentPrecisionWidgetIndex !== -1) {
                    node.widgets.splice(currentPrecisionWidgetIndex, 1);
                    fitHeight(node); // Adjust height immediately after hiding
                }
            }
            updateModelLists(); // Call update after provider changes
        };

        const origPrecisionCallback = precisionWidget.callback;
        precisionWidget.callback = (value) => {
            origPrecisionCallback?.call(node, value);
            updateModelLists(); // Call update after precision changes
        };

        // --- Conditional visibility for corner_position widget ---
        const origNumberedCompositeCallback = numberedCompositeWidget.callback;
        numberedCompositeWidget.callback = (value) => {
            origNumberedCompositeCallback?.call(node, value);

            if (value === true) {
                // If the corner_position widget is currently removed, re-insert it
                if (node.widgets.findIndex(w => w.name === "corner_position") === -1) {
                    node.widgets.splice(node.widgets.findIndex(w => w.name === "numbered_composite") + 1, 0, cornerPositionWidgetReference);
                    fitHeight(node); // Adjust height immediately after showing
                }
            } else { // Condition to hide corner_position widget
                // If the corner_position widget is currently present, remove it
                const currentCornerPositionWidgetIndex = node.widgets.findIndex(w => w.name === "corner_position");
                if (currentCornerPositionWidgetIndex !== -1) {
                    node.widgets.splice(currentCornerPositionWidgetIndex, 1);
                    fitHeight(node); // Adjust height immediately after hiding
                }
            }
        };

        // Set initial visibility of corner_position widget
        if (numberedCompositeWidget.value === false) {
            const currentCornerPositionWidgetIndex = node.widgets.findIndex(w => w.name === "corner_position");
            if (currentCornerPositionWidgetIndex !== -1) {
                node.widgets.splice(currentCornerPositionWidgetIndex, 1);
            }
        }
        fitHeight(node); // Adjust height after initial visibility setup

        // Add a delayed call to handle all UI updates after startup.
        setTimeout(() => {
            console.log("[DWposeNode] Performing delayed UI refresh for provider and models.");

            // Per user suggestion, explicitly handle precision widget visibility here.
            const providerType = providerWidget.value;
            if (providerType === "GPU") {
                // If the precision widget is currently removed, re-insert it
                if (node.widgets.findIndex(w => w.name === "precision") === -1) {

                    precisionWidgetReference.value = "fp32"; 
                    
                    node.widgets.splice(node.widgets.findIndex(w => w.name === "provider_type") + 1, 0, precisionWidgetReference);
                    fitHeight(node);
                }
            } else {
                // If the precision widget is currently present, remove it
                const currentPrecisionWidgetIndex = node.widgets.findIndex(w => w.name === "precision");
                if (currentPrecisionWidgetIndex !== -1) {
                    node.widgets.splice(currentPrecisionWidgetIndex, 1);
                    fitHeight(node);
                }
            }

            // This will refresh the model list, which is the original purpose of the delay.
            updateModelLists();
        }, 1000);

        // Add a listener for when the node's execute method completes
        node.onExecuted = async function(message) {
            console.log("[DWposeNode] onExecuted triggered. Message:", message);
            let refreshNeeded = false;
            if (Array.isArray(message?.ui)) {
                for (const ui_item of message.ui) {
                    console.log("[DWposeNode] ui_item being processed:", ui_item);
                    if (ui_item === "model_refresh_needed") {
                        refreshNeeded = true;
                        break;
                    }
                }
            }

            if (refreshNeeded) {
                console.log("[DWposeNode] Model refresh needed signal received. Updating model lists...");
                // Programmatically "flick" the provider_type widget to trigger a refresh
                const originalProviderValue = providerWidget.value;
                // Temporarily change to a different value, then back to original
                providerWidget.value = (originalProviderValue === "CPU") ? "GPU" : "CPU";
                providerWidget.callback(providerWidget.value);
                providerWidget.value = originalProviderValue;
                providerWidget.callback(originalProviderValue);
                await updateModelLists();
            } else {
                console.log("[DWposeNode] Model refresh not needed or signal not found in message.ui.");
            }
        };
    }
});