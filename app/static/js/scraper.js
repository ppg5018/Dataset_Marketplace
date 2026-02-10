// ==========================================
// WEB SCRAPER FUNCTIONALITY
// ==========================================

let selectedMethod = 'table';
let scrapedData = null;

// Method Selection
function selectMethod(method) {
    selectedMethod = method;

    // Update UI
    document.querySelectorAll('.method-card').forEach(card => {
        card.classList.remove('active');
    });
    document.querySelector(`[data-method="${method}"]`).classList.add('active');

    // Show/hide custom selectors
    if (method === 'custom') {
        document.getElementById('custom-selectors').style.display = 'block';
    } else {
        document.getElementById('custom-selectors').style.display = 'none';
    }
}

// Add Custom Selector Input
function addSelector() {
    const container = document.getElementById('selector-inputs');
    const newRow = document.createElement('div');
    newRow.style.cssText = 'display: flex; gap: 1rem; margin-bottom: 0.5rem;';
    newRow.innerHTML = `
        <input type="text" placeholder="Column Name" class="selector-name"
            style="flex: 1; padding: 0.75rem; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-secondary);">
        <input type="text" placeholder="CSS Selector" class="selector-value"
            style="flex: 2; padding: 0.75rem; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-secondary);">
        <button onclick="this.parentElement.remove()" class="btn btn-secondary" style="padding: 0.75rem;">
            <i class="fas fa-times"></i>
        </button>
    `;
    container.appendChild(newRow);
}

// Start Scraping
function startScrape() {
    const url = document.getElementById('scrape-url').value.trim();
    if (!url) {
        showToast('Please enter a URL', 'error');
        return;
    }

    // Get selectors if custom method
    let selectors = {};
    if (selectedMethod === 'custom') {
        const names = document.querySelectorAll('.selector-name');
        const values = document.querySelectorAll('.selector-value');

        for (let i = 0; i < names.length; i++) {
            const name = names[i].value.trim();
            const value = values[i].value.trim();
            if (name && value) {
                selectors[name] = value;
            }
        }

        if (Object.keys(selectors).length === 0) {
            showToast('Please add at least one selector', 'error');
            return;
        }
    }

    // Update status
    const statusEl = document.getElementById('scrape-status');
    statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scraping...</span>';
    statusEl.style.color = 'white';

    // Send request
    fetch('/scraper/api/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            url: url,
            method: selectedMethod,
            selectors: Object.keys(selectors).length > 0 ? selectors : null
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                statusEl.innerHTML = '<i class="fas fa-check"></i> Success!';
                statusEl.style.color = '#10b981';

                scrapedData = data;
                displayPreview(data);

                // Auto-fill title
                try {
                    const urlObj = new URL(url);
                    document.getElementById('dataset-title').value =
                        `Data from ${urlObj.hostname}`;
                } catch (e) { }

            } else {
                statusEl.innerHTML = '<i class="fas fa-times"></i> Error';
                statusEl.style.color = '#ef4444';
                showToast(data.error || 'Scraping failed', 'error');
            }
        })
        .catch(err => {
            statusEl.innerHTML = '<i class="fas fa-times"></i> Error';
            statusEl.style.color = '#ef4444';
            showToast('Network error: ' + err.message, 'error');
        });
}

// Display Preview
function displayPreview(data) {
    const previewSection = document.getElementById('preview-section');
    const statsEl = document.getElementById('preview-stats');
    const tableEl = document.getElementById('preview-table');

    previewSection.style.display = 'block';

    // Stats
    statsEl.innerHTML = `
        <span style="margin-right: 2rem;"><i class="fas fa-table"></i> <strong>${data.columns.length}</strong> columns</span>
        <span><i class="fas fa-database"></i> <strong>${data.rows}</strong> rows</span>
    `;

    // Table
    if (data.preview && data.preview.length > 0) {
        let tableHTML = '<table><thead><tr>';

        // Headers
        data.columns.forEach(col => {
            tableHTML += `<th>${col}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';

        // Rows (first 50)
        data.preview.slice(0, 50).forEach(row => {
            tableHTML += '<tr>';
            data.columns.forEach(col => {
                const value = row[col] || '';
                tableHTML += `<td>${String(value).substring(0, 100)}</td>`;
            });
            tableEl += '</tr>';
        });

        tableHTML += '</tbody></table>';
        tableEl.innerHTML = tableHTML;
    } else {
        tableEl.innerHTML = '<p style="padding: 2rem; text-align: center; color: var(--text-secondary);">No data preview available</p>';
    }

    // Scroll to preview
    previewSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Save Dataset
function saveDataset() {
    const url = document.getElementById('scrape-url').value.trim();
    const title = document.getElementById('dataset-title').value.trim();
    const description = document.getElementById('dataset-description').value.trim();
    const price = parseFloat(document.getElementById('dataset-price').value) || 0;

    if (!title) {
        showToast('Please enter a title', 'error');
        return;
    }

    if (!scrapedData) {
        showToast('No data to save', 'error');
        return;
    }

    // Get selectors if custom method
    let selectors = {};
    if (selectedMethod === 'custom') {
        const names = document.querySelectorAll('.selector-name');
        const values = document.querySelectorAll('.selector-value');

        for (let i = 0; i < names.length; i++) {
            const name = names[i].value.trim();
            const value = values[i].value.trim();
            if (name && value) {
                selectors[name] = value;
            }
        }
    }

    showToast('Saving dataset...', 'info');

    fetch('/scraper/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            url: url,
            method: selectedMethod,
            selectors: Object.keys(selectors).length > 0 ? selectors : null,
            title: title,
            description: description,
            price: price
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('Dataset saved successfully!', 'success');

                // Confetti!
                if (typeof createConfetti === 'function') {
                    createConfetti();
                }

                // Redirect to dataset
                setTimeout(() => {
                    window.location.href = `/dataset/${data.dataset_id}`;
                }, 2000);
            } else {
                showToast(data.error || 'Failed to save dataset', 'error');
            }
        })
        .catch(err => {
            showToast('Error: ' + err.message, 'error');
        });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Select default method
    selectMethod('table');
});
