document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form values
    const formData = {
        average_cost: document.getElementById('average_cost').value,
        rating: document.getElementById('rating').value,
        votes: document.getElementById('votes').value,
        location: document.getElementById('location').value,
        cuisines: document.getElementById('cuisines').value
    };
    
    try {
        // Show loading state
        const submitButton = e.target.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.textContent;
        submitButton.textContent = 'Predicting...';
        submitButton.disabled = true;
        
        // Make API call
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Show prediction
            document.getElementById('predictionResult').style.display = 'block';
            document.getElementById('predictionValue').textContent = 
                `Estimated delivery time: ${data.predicted_delivery_time} minutes`;
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction. Please try again.');
    } finally {
        // Restore button state
        submitButton.textContent = originalButtonText;
        submitButton.disabled = false;
    }
}); 