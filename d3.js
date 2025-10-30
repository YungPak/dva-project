document.getElementById('movieForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = {
        title: document.getElementById('title').value,
        genre: document.getElementById('genre').value,
        budget: parseFloat(document.getElementById('budget').value),
        director: document.getElementById('director').value,
        cast: document.getElementById('cast').value,
        releaseMonth: parseInt(document.getElementById('releaseMonth').value),
        runtime: parseInt(document.getElementById('runtime').value)
    };

    makePrediction(formData);
});

function makePrediction(data) {
    const mockPrediction = (data.budget * 2.5) + (Math.random() * 50);
    
    document.getElementById('predictionAmount').textContent = 
        '$' + mockPrediction.toFixed(2) + 'M';
    document.getElementById('predictionResult').classList.add('show');
    
    updateD3Visualization(data, mockPrediction);
}

function updateD3Visualization(inputData, prediction) {
    document.getElementById('d3-chart').innerHTML = 
        '<p>D3 Chart: Feature importance visualization goes here</p>' +
        '<p style="font-size: 0.9em; margin-top: 10px;">Shows how each factor (budget, genre, cast, etc.) influences the prediction</p>';
    
}

function resetForm() {
    document.getElementById('movieForm').reset();
    document.getElementById('predictionResult').classList.remove('show');
    document.getElementById('d3-chart').innerHTML = 
        'D3.js Visualization will appear here after prediction';
}