document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('toggle-btn');
    const statusOverlay = document.getElementById('inference-status');
    let isInferenceEnabled = false;

    toggleBtn.addEventListener('click', async () => {
        // Toggle optimistic state
        const newState = !isInferenceEnabled;
        
        try {
            const response = await fetch('/api/inference', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ enabled: newState })
            });

            if (response.ok) {
                const data = await response.json();
                isInferenceEnabled = data.inference_enabled;
                
                // Update UI based on confirmed state
                if (isInferenceEnabled) {
                    toggleBtn.textContent = 'Stop Inference';
                    toggleBtn.classList.remove('btn-primary');
                    toggleBtn.classList.add('btn-danger');
                    statusOverlay.textContent = 'Inference: ON';
                    statusOverlay.classList.add('active');
                } else {
                    toggleBtn.textContent = 'Start Inference';
                    toggleBtn.classList.remove('btn-danger');
                    toggleBtn.classList.add('btn-primary');
                    statusOverlay.textContent = 'Inference: OFF';
                    statusOverlay.classList.remove('active');
                }
            } else {
                console.error('Failed to toggle inference');
                alert('Error communcating with the server.');
            }
        } catch (err) {
            console.error(err);
        }
    });
});
