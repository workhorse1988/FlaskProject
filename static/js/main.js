document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const resultSection = document.getElementById('resultSection');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorAlert = document.getElementById('errorAlert');

    const MAX_FILE_SIZE = 526 * 1024 * 1024; // 526MB in bytes

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        if (!fileInput.files[0]) {
            showError('Please select a file to upload');
            return;
        }

        // Check file size before uploading
        if (fileInput.files[0].size > MAX_FILE_SIZE) {
            showError(`File size exceeds the limit of 526MB. Please choose a smaller file.`);
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            showLoading(true);
            hideError();

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to process file');
            }

            displayResults(data);
        } catch (error) {
            showError(error.message);
        } finally {
            showLoading(false);
        }
    });

    function displayResults(data) {
        resultSection.innerHTML = `
            <div class="card mb-4">
                <div class="card-header">
                    <h4 class="mb-0">Analysis Results</h4>
                </div>
                <div class="card-body">
                    <h5>Document Type: ${data.document_type}</h5>
                    <h6>System Type: ${data.analysis.system_type}</h6>
                    <h6>Occupancy Classification: ${data.analysis.occupancy_classification}</h6>

                    <h5 class="mt-4">System Design</h5>
                    <ul class="list-group">
                        <li class="list-group-item">Density Requirement: ${data.analysis.system_design.density_requirement}</li>
                        <li class="list-group-item">Total Water Demand: ${data.analysis.system_design.total_water_demand}</li>
                        <li class="list-group-item">Sprinkler Spacing: ${data.analysis.system_design.sprinkler_spacing}</li>
                    </ul>

                    <h5 class="mt-4">Potential Discrepancies</h5>
                    <div class="alert alert-warning">
                        ${data.analysis.potential_discrepancies.map(d => `
                            <div class="mb-2">
                                <strong>Issue:</strong> ${d.issue}<br>
                                <strong>Reference:</strong> ${d.compliance_reference}
                            </div>
                        `).join('')}
                    </div>

                    <a href="/download-report/fire_sprinkler_analysis.json" class="btn btn-primary">
                        Download Full Report
                    </a>
                </div>
            </div>
        `;
        resultSection.style.display = 'block';
    }

    function showLoading(show) {
        loadingSpinner.style.display = show ? 'block' : 'none';
    }

    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.style.display = 'block';
    }

    function hideError() {
        errorAlert.style.display = 'none';
    }
});