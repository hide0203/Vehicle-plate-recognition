// Main JavaScript for the plate recognition system

class PlateRecognitionApp {
    constructor() {
        this.isProcessing = false;
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.startPeriodicUpdates();
    }
    
    setupEventListeners() {
        // File upload drag and drop
        const uploadZone = document.getElementById('uploadZone');
        if (uploadZone) {
            uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadZone.addEventListener('drop', this.handleDrop.bind(this));
            uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        }
        
        // Form submission
        const uploadForm = document.getElementById('uploadForm');
        if (uploadForm) {
            uploadForm.addEventListener('submit', this.handleFormSubmit.bind(this));
        }
        
        // Search functionality
        const searchForm = document.getElementById('searchForm');
        if (searchForm) {
            searchForm.addEventListener('submit', this.handleSearch.bind(this));
        }
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.currentTarget.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processImage(files[0]);
        }
    }
    
    handleFormSubmit(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('imageFile');
        if (fileInput.files.length > 0) {
            this.processImage(fileInput.files[0]);
        }
    }
    
    processImage(file) {
        if (this.isProcessing) return;
        
        this.isProcessing = true;
        this.showLoading();
        
        const formData = new FormData();
        formData.append('image', file);
        
        fetch('/api/upload_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            this.displayResults(data);
        })
        .catch(error => {
            this.displayError('Error processing image: ' + error.message);
        })
        .finally(() => {
            this.isProcessing = false;
            this.hideLoading();
        });
    }
    
    displayResults(data) {
        const resultDiv = document.getElementById('uploadResult');
        
        if (data.success) {
            let html = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i> 
                    Found ${data.plates_detected} plate(s)
                </div>
            `;
            
            if (data.results && data.results.length > 0) {
                html += '<div class="row">';
                data.results.forEach((result, index) => {
                    const confidenceClass = result.confidence > 80 ? 'success' : 
                                          result.confidence > 60 ? 'warning' : 'danger';
                    
                    html += `
                        <div class="col-md-6 mb-3">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">${result.plate_number}</h5>
                                    <p class="card-text">
                                        <span class="badge bg-${confidenceClass}">
                                            Confidence: ${result.confidence.toFixed(1)}%
                                        </span>
                                    </p>
                                    <small class="text-muted">
                                        Coordinates: ${result.coordinates.join(', ')}
                                    </small>
                                </div>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
            }
            
            resultDiv.innerHTML = html;
        } else {
            this.displayError(data.error || 'Unknown error occurred');
        }
    }
    
    displayError(message) {
        const resultDiv = document.getElementById('uploadResult');
        resultDiv.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> 
                ${message}
            </div>
        `;
    }
    
    showLoading() {
        const resultDiv = document.getElementById('uploadResult');
        resultDiv.innerHTML = `
            <div class="text-center p-3">
                <div class="spinner"></div>
                <p class="mt-2">Processing image...</p>
            </div>
        `;
    }
    
    hideLoading() {
        // Loading will be replaced by results or error
    }
    
    startPeriodicUpdates() {
        // Update statistics every 30 seconds
        setInterval(() => {
            this.updateStatistics();
        }, 30000);
        
        // Update recent plates every 10 seconds
        setInterval(() => {
            this.updateRecentPlates();
        }, 10000);
    }
    
    updateStatistics() {
        fetch('/api/statistics')
        .then(response => response.json())
        .then(data => {
            this.updateStatsDisplay(data);
        })
        .catch(error => {
            console.log('Error updating statistics:', error);
        });
    }
    
    updateStatsDisplay(stats) {
        const totalElement = document.getElementById('totalPlates');
        const uniqueElement = document.getElementById('uniquePlates');
        const todayElement = document.getElementById('todayDetections');
        
        if (totalElement) totalElement.textContent = stats.total_plates;
        if (uniqueElement) uniqueElement.textContent = stats.unique_plates;
        if (todayElement) todayElement.textContent = stats.today_detections;
    }
    
    updateRecentPlates() {
        fetch('/api/recent_plates?limit=10')
        .then(response => response.json())
        .then(data => {
            this.updateRecentPlatesDisplay(data);
        })
        .catch(error => {
            console.log('Error updating recent plates:', error);
        });
    }
    
    updateRecentPlatesDisplay(plates) {
        const container = document.getElementById('recentPlatesContainer');
        if (!container) return;
        
        let html = '';
        plates.forEach(plate => {
            const date = new Date(plate.timestamp);
            const timeStr = date.toLocaleTimeString();
            
            html += `
                <div class="d-flex justify-content-between align-items-center mb-2 p-2 border-bottom">
                    <div>
                        <strong>${plate.plate_number}</strong><br>
                        <small class="text-muted">${timeStr}</small>
                    </div>
                    ${plate.confidence_score ? 
                        `<span class="badge bg-secondary">${plate.confidence_score.toFixed(1)}%</span>` 
                        : ''
                    }
                </div>
            `;
        });
        
        container.innerHTML = html || '<p class="text-muted text-center">No recent detections</p>';
    }
    
    handleSearch(e) {
        e.preventDefault();
        
        const query = document.getElementById('searchQuery').value;
        if (query.trim()) {
            window.location.href = `/search?q=${encodeURIComponent(query)}`;
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    new PlateRecognitionApp();
});

// Utility functions
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}
