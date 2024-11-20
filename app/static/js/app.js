// app/static/js/app.js

document.addEventListener('DOMContentLoaded', function() {
    // Form submission handling
    const form = document.querySelector('.settings-form');
    const loadingOverlay = document.querySelector('.loading');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            loadingOverlay.classList.add('active');
        });
    }
    
    // Model selection handling
    const modelRadios = document.querySelectorAll('input[name="model_name"]');
    modelRadios.forEach(radio => {
        radio.addEventListener('change', async function() {
            const modelInfo = await fetch(`/api/model_info/${this.value}`).then(r => r.json());
            if (modelInfo.success) {
                // Update UI with model information
                updateModelInfo(modelInfo.info);
            }
        });
    });
    
    // Helper function to update model info in UI
    function updateModelInfo(info) {
        const infoContainer = document.querySelector('.model-info');
        if (infoContainer) {
            infoContainer.innerHTML = `
                <h3>${info.name}</h3>
                <p>${info.description}</p>
            `;
        }
    }
    
    // Copy results to clipboard
    const copyButtons = document.querySelectorAll('.copy-results');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const textToCopy = this.dataset.text;
            navigator.clipboard.writeText(textToCopy)
                .then(() => {
                    button.textContent = 'Copied!';
                    setTimeout(() => {
                        button.textContent = 'Copy';
                    }, 2000);
                });
        });
    });
});