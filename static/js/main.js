document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
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

        const response = await fetch('/analyze', {
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
            resultAlert.textContent = `Error: ${data.detail || data.message || 'Unknown error'}`;
        }
    } catch (error) {
        loading.style.display = 'none';
        results.style.display = 'block';
        resultAlert.className = 'alert alert-danger';
        resultAlert.textContent = 'Error processing the request';
        console.error('Error:', error);
    }
}); 