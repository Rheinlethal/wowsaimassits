const form = document.getElementById('predictForm');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');
const offsetValueSpan = document.getElementById('offsetValue');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
    
    const formData = new FormData(form);
    const data = {
        shell_travel_time: parseFloat(formData.get('shell_travel_time')),
        distance: parseFloat(formData.get('distance')),
        angle: parseFloat(formData.get('angle'))
    };
    
    if (isNaN(data.shell_travel_time) || isNaN(data.distance) || isNaN(data.angle)) {
        showError('Please enter valid numbers for all fields');
        return;
    }
    
    if (data.shell_travel_time <= 0 || data.distance <= 0) {
        showError('Shell travel time and distance must be positive values');
        return;
    }
    
    const button = form.querySelector('button');
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="btn-icon">⏳</span> Calculating...';
    button.disabled = true;
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        offsetValueSpan.textContent = result.offset_x;
        resultDiv.classList.remove('hidden');
        
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
    } catch (error) {
        showError(`Error: ${error.message}. Please try again.`);
    } finally {
        button.innerHTML = originalText;
        button.disabled = false;
    }
});

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
