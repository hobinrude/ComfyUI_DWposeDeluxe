import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const style = document.createElement('style');
style.textContent = `
    .dwpose-modal { position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.6); }
    .dwpose-modal-content { background-color: #282828; color: #f0f0f0; margin: 10% auto; padding: 20px; border: 1px solid #555; width: 80%; max-width: 800px; box-shadow: 0 5px 15px rgba(0,0,0,0.5); }
    .dwpose-modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #444; }
    .dwpose-modal-listings { max-height: 50vh; overflow-y: auto; }
    .dwpose-modal-item { padding: 8px; cursor: pointer; border-radius: 3px; }
    .dwpose-modal-item:hover { background-color: #4a4a4a; }
    .dwpose-modal-dir { color: #87ceeb; }
    .dwpose-modal-file { color: #f0f0f0; }
`;
document.head.appendChild(style);

app.registerExtension({
    name: "dwpose_adv.fileBrowserAndDropdown",
    async nodeCreated(node) {
        if (node.comfyClass === "LoadPoseKeypoints") {
            const fileWidget = node.widgets.find(w => w.name === "file");

            // Function to update the file list dropdown
            const updateFileList = async (selectedFilename = null) => {
                const response = await fetch(`/dwpose_adv/get_files`);
                const files = await response.json();

                // Ensure "[none]" is always an option if no files are found
                if (files.length === 0) {
                    files = ["[none]"];
                }

                fileWidget.options.values = files;
                // Select the uploaded file if available, otherwise select the current value or the first item
                if (selectedFilename && files.includes(selectedFilename)) {
                    fileWidget.value = selectedFilename;
                } else if (!files.includes(fileWidget.value)) {
                   fileWidget.value = files[0];
                }
                app.graph.setDirtyCanvas(true, true);
            };

            // Helper function to upload files (similar to VHS.core.js)
            async function uploadFile(file, progressCallback) {
                try {
                    const body = new FormData();
                    body.append("image", file); // ComfyUI's API uses 'image' for file uploads, even for JSON
                    const url = api.apiURL("/upload/image");

                    const resp = await new Promise((resolve, reject) => {
                        let req = new XMLHttpRequest();
                        req.upload.onprogress = (e) => progressCallback?.(e.loaded / e.total);
                        req.onload = () => resolve(req);
                        req.onerror = () => reject(new Error("Network error"));
                        req.open('post', url, true);
                        req.send(body);
                    });

                    if (resp.status !== 200) {
                        throw new Error(`${resp.status} - ${resp.statusText}`);
                    }
                    return resp;
                } catch (error) {
                    alert(`Upload error: ${error.message}`);
                    throw error;
                }
            }

            // Add the "Upload keypoint dataset" button
            const uploadButton = node.addWidget("button", "Upload keypoint dataset", "upload", async () => {
                // Create a hidden file input element
                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = ".json"; // Accept only JSON files
                fileInput.style.display = "none";
                document.body.appendChild(fileInput);

                fileInput.onchange = async (e) => {
                    if (fileInput.files.length > 0) {
                        try {
                            const uploadedFile = fileInput.files[0];
                            // Show progress (optional)
                            node.progress = 0; // Initialize progress
                            const resp = await uploadFile(uploadedFile, (p) => { node.progress = p; });
                            node.progress = undefined; // Clear progress

                            const filename = JSON.parse(resp.responseText).name; // Get the filename from the API response
                            
                            // Refresh the file list and select the new file
                            await updateFileList(filename);

                        } catch (error) {
                            console.error("Error during upload:", error);
                        }
                    }
                    document.body.removeChild(fileInput); // Clean up the hidden input
                };

                // Programmatically click the hidden file input
                fileInput.click();
            });
            uploadButton.options.serialize = false; // Don't save this button in workflow

            // Initial file list load
            setTimeout(() => {
                updateFileList();
            }, 1);
        }
    }
});