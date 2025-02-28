document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const API_URL = window.location.origin;  // This will work for both local and production
    const fileInput = document.getElementById('eegFile');
    const file = fileInput.files[0];
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultAlert = document.getElementById('resultAlert');

    if (!file) {
        alert('Please select a file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        loading.style.display = 'block';
        results.style.display = 'none';

        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        
        loading.style.display = 'none';
        results.style.display = 'block';

        if (data.status === 'success') {
            resultAlert.className = 'alert alert-success';
            resultAlert.textContent = `Analysis Result: ${data.message}`;
        } else {
            resultAlert.className = 'alert alert-danger';
            resultAlert.textContent = `Error: ${data.detail || 'Error processing the file. Please try again.'}`;
        }
    } catch (error) {
        loading.style.display = 'none';
        results.style.display = 'block';
        resultAlert.className = 'alert alert-danger';
        resultAlert.textContent = 'Network error or server unavailable. Please try again later.';
        console.error('Error:', error);
    }
}); 