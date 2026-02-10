document.addEventListener('DOMContentLoaded', function () {
    // Tab switching logic
    window.showTab = function (tabId) {
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });

        // Show selected tab
        document.getElementById(tabId).classList.add('active');
        event.currentTarget.classList.add('active');
    };

    // Load preview if on detail page and has access
    if (typeof DATASET_ID !== 'undefined') {
        loadPreview(DATASET_ID);
        populateColumnSelects();
    }
});

function loadPreview(datasetId) {
    fetch(`/api/data/${datasetId}/preview`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('preview-table-container').innerHTML = `<p class="error">${data.error}</p>`;
                return;
            }
            renderTable(data, 'preview-table-container');
        })
        .catch(err => console.error(err));
}

function renderTable(data, containerId) {
    if (!data || data.length === 0) {
        document.getElementById(containerId).innerHTML = '<p>No data available</p>';
        return;
    }

    const headers = Object.keys(data[0]);
    let html = '<table><thead><tr>';
    headers.forEach(h => html += `<th>${h}</th>`);
    html += '</tr></thead><tbody>';

    data.forEach(row => {
        html += '<tr>';
        headers.forEach(h => html += `<td>${row[h]}</td>`);
        html += '</tr>';
    });
    html += '</tbody></table>';

    document.getElementById(containerId).innerHTML = html;
}

function populateColumnSelects() {
    if (typeof COLUMNS === 'undefined') return;

    const targetSelect = document.getElementById('target-col');
    const featureSelect = document.getElementById('feature-cols');
    const vizXSelect = document.getElementById('viz-x-col');
    const vizYSelect = document.getElementById('viz-y-col');
    const outlierCols = document.getElementById('outlier-cols');
    const transformCol = document.getElementById('transform-col');
    const boxplotCol = document.getElementById('boxplot-col');

    COLUMNS.forEach(col => {
        // Target
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        targetSelect.appendChild(option);

        // Features
        const featureOption = document.createElement('option');
        featureOption.value = col;
        featureOption.textContent = col;
        featureSelect.appendChild(featureOption);

        // Viz X
        const vizXOption = document.createElement('option');
        vizXOption.value = col;
        vizXOption.textContent = col;
        vizXSelect.appendChild(vizXOption);

        // Viz Y
        const vizYOption = document.createElement('option');
        vizYOption.value = col;
        vizYOption.textContent = col;
        vizYSelect.appendChild(vizYOption);

        // Outlier columns
        if (outlierCols) {
            const outlierOption = document.createElement('option');
            outlierOption.value = col;
            outlierOption.textContent = col;
            outlierCols.appendChild(outlierOption);
        }

        // Transform column
        if (transformCol) {
            const transformOption = document.createElement('option');
            transformOption.value = col;
            transformOption.textContent = col;
            transformCol.appendChild(transformOption);
        }

        // Boxplot column
        if (boxplotCol) {
            const boxplotOption = document.createElement('option');
            boxplotOption.value = col;
            boxplotOption.textContent = col;
            boxplotCol.appendChild(boxplotOption);
        }
    });
}

window.runCleaning = function (datasetId) {
    const operations = [];
    if (document.getElementById('drop-na').checked) {
        operations.push({ type: 'drop_na' });
    }
    if (document.getElementById('drop-dup').checked) {
        operations.push({ type: 'drop_duplicates' });
    }

    document.getElementById('clean-preview-container').innerHTML = 'Processing...';

    fetch(`/api/data/${datasetId}/clean`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ operations: operations })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('clean-preview-container').innerHTML = `<p class="error">${data.error}</p>`;
            } else {
                renderTable(data.preview, 'clean-preview-container');
            }
        });
};

let chartInstance = null;

window.runVisualization = function (datasetId) {
    const xCol = document.getElementById('viz-x-col').value;
    const yCol = document.getElementById('viz-y-col').value;
    const chartType = document.getElementById('viz-type').value;

    fetch(`/api/data/${datasetId}/visualize?x_col=${xCol}&y_col=${yCol}&chart_type=${chartType}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            const ctx = document.getElementById('viz-canvas').getContext('2d');
            if (chartInstance) {
                chartInstance.destroy();
            }

            let chartData = {};
            let options = { responsive: true, maintainAspectRatio: false };

            if (chartType === 'bar') {
                const labels = data.map(d => d[xCol]);
                const values = data.map(d => d.count);
                chartData = {
                    labels: labels,
                    datasets: [{
                        label: 'Count',
                        data: values,
                        backgroundColor: 'rgba(99, 102, 241, 0.5)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 1
                    }]
                };
            } else if (chartType === 'scatter') {
                const values = data.map(d => ({ x: d[xCol], y: d[yCol] }));
                chartData = {
                    datasets: [{
                        label: `${yCol} vs ${xCol}`,
                        data: values,
                        backgroundColor: 'rgba(99, 102, 241, 0.5)'
                    }]
                };
                options.scales = {
                    x: { type: 'linear', position: 'bottom', title: { display: true, text: xCol } },
                    y: { title: { display: true, text: yCol } }
                };
            } else if (chartType === 'histogram') {
                // Simple histogram visualization (assuming pre-binned or raw values)
                // For raw values, we might need a histogram plugin or binning on client
                // Here we just plot raw values as a line for simplicity or assume backend returns bins
                // Since backend returns raw list for histogram in our impl, let's just warn
                alert("Histogram requires binning logic. Showing raw values.");
                return;
            }

            chartInstance = new Chart(ctx, {
                type: chartType === 'histogram' ? 'bar' : chartType,
                data: chartData,
                options: options
            });
        });
};

window.runML = function (datasetId) {
    const targetCol = document.getElementById('target-col').value;
    const featureCols = Array.from(document.getElementById('feature-cols').selectedOptions).map(opt => opt.value);
    const modelType = document.getElementById('model-type').value;

    if (!targetCol || featureCols.length === 0) {
        alert('Please select a target column and at least one feature column.');
        return;
    }

    document.getElementById('ml-results').innerHTML = 'Training model...';

    fetch(`/api/data/${datasetId}/ml`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            target_col: targetCol,
            feature_cols: featureCols,
            model_type: modelType
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('ml-results').innerHTML = `<p class="error">${data.error}</p>`;
            } else {
                document.getElementById('ml-results').innerHTML = `
                <div class="alert" style="background-color: rgba(16, 185, 129, 0.1); border-color: var(--success-color); color: var(--success-color);">
                    <h4>Training Complete</h4>
                    <p>Model Type: <strong>${data.model_type}</strong></p>
                    <p>${data.metric}: <strong>${data.value.toFixed(4)}</strong></p>
                </div>
            `;

                // Show prediction section
                document.getElementById('prediction-section').style.display = 'block';
                const inputsContainer = document.getElementById('prediction-inputs');
                inputsContainer.innerHTML = '';
                featureCols.forEach(col => {
                    const div = document.createElement('div');
                    div.className = 'form-group';
                    div.innerHTML = `<label>${col}</label><input type="number" step="any" class="pred-input" data-col="${col}">`;
                    inputsContainer.appendChild(div);
                });
            }
        });
};

window.runPrediction = function (datasetId) {
    const inputs = {};
    document.querySelectorAll('.pred-input').forEach(input => {
        inputs[input.dataset.col] = parseFloat(input.value);
    });

    const featureCols = Array.from(document.getElementById('feature-cols').selectedOptions).map(opt => opt.value);

    fetch(`/api/data/${datasetId}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            input_data: inputs,
            feature_cols: featureCols
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('prediction-result').innerText = `Error: ${data.error}`;
            } else {
                document.getElementById('prediction-result').innerText = `Prediction: ${data.prediction}`;
            }
        });
};

window.downloadCleaned = function (datasetId) {
    const operations = [];
    if (document.getElementById('drop-na').checked) {
        operations.push({ type: 'drop_na' });
    }
    if (document.getElementById('drop-dup').checked) {
        operations.push({ type: 'drop_duplicates' });
    }

    fetch(`/api/data/${datasetId}/download_cleaned`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ operations: operations })
    })
        .then(response => {
            if (response.ok) {
                return response.blob();
            }
            throw new Error('Download failed');
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cleaned_dataset_${datasetId}.csv`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        })
        .catch(err => alert(err.message));
};

window.downloadChart = function (canvasId) {
    const canvas = document.getElementById(canvasId);
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = 'chart.png';
    document.body.appendChild(a);
    a.click();
    a.remove();
};

window.runAutoViz = function (datasetId) {
    const container = document.getElementById('auto-viz-container');
    container.innerHTML = 'Generating plots...';

    fetch(`/api/data/${datasetId}/auto_visualize`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = `<p class="error">${data.error}</p>`;
                return;
            }

            container.innerHTML = '';

            data.forEach((viz, index) => {
                const card = document.createElement('div');
                card.className = 'card';
                card.style.padding = '1rem';

                const title = document.createElement('h4');
                title.textContent = `${viz.column} (${viz.type})`;
                card.appendChild(title);

                const canvasContainer = document.createElement('div');
                canvasContainer.style.position = 'relative';
                canvasContainer.style.height = '250px';

                const canvas = document.createElement('canvas');
                canvas.id = `auto-viz-${index}`;
                canvasContainer.appendChild(canvas);
                card.appendChild(canvasContainer);

                const downloadBtn = document.createElement('button');
                downloadBtn.className = 'btn btn-sm btn-secondary';
                downloadBtn.textContent = 'Download';
                downloadBtn.style.marginTop = '0.5rem';
                downloadBtn.onclick = () => downloadChart(canvas.id);
                card.appendChild(downloadBtn);

                container.appendChild(card);

                // Render Chart
                const ctx = canvas.getContext('2d');
                let chartData = {};
                let options = { responsive: true, maintainAspectRatio: false };

                if (viz.type === 'bar') {
                    const labels = viz.data.map(d => d[viz.column]);
                    const values = viz.data.map(d => d.count);
                    chartData = {
                        labels: labels,
                        datasets: [{
                            label: 'Count',
                            data: values,
                            backgroundColor: 'rgba(16, 185, 129, 0.5)',
                            borderColor: 'rgba(16, 185, 129, 1)',
                            borderWidth: 1
                        }]
                    };
                } else if (viz.type === 'histogram') {
                    // Simple histogram visualization (raw values)
                    const values = viz.data[viz.column];
                    // We'll just plot them as a bar chart of raw values for now, or use a scatter
                    // A proper histogram needs binning. Let's simulate bins or just plot raw.
                    // For simplicity in this MVP, we plot raw values as a line chart
                    const labels = values.map((_, i) => i);
                    chartData = {
                        labels: labels,
                        datasets: [{
                            label: 'Value',
                            data: values,
                            backgroundColor: 'rgba(16, 185, 129, 0.5)',
                            type: 'line'
                        }]
                    };
                }

                new Chart(ctx, {
                    type: viz.type === 'histogram' ? 'line' : 'bar',
                    data: chartData,
                    options: options
                });
            });
        });
};

window.generateSynthetic = function (datasetId) {
    const numRows = document.getElementById('synthetic-rows').value;

    fetch(`/api/data/${datasetId}/generate_synthetic`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ num_rows: numRows })
    })
        .then(response => {
            if (response.ok) {
                return response.blob();
            }
            return response.json().then(err => { throw new Error(err.error || 'Error generating data') });
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `synthetic_dataset_${datasetId}.csv`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        })
        .catch(err => alert(err.message));
};

// Advanced Data Engineering
window.runOutlierRemoval = function (datasetId) {
    const columns = Array.from(document.getElementById('outlier-cols').selectedOptions).map(opt => opt.value);
    const method = document.getElementById('outlier-method').value;
    const threshold = document.getElementById('outlier-threshold').value;

    if (columns.length === 0) {
        alert('Please select at least one column');
        return;
    }

    document.getElementById('advanced-preview-container').innerHTML = 'Processing...';

    fetch(`/api/data/${datasetId}/clean/advanced`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: 'outliers', columns, method, threshold })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('advanced-preview-container').innerHTML = `<p class="error">${data.error}</p>`;
            } else {
                renderTable(data.preview, 'advanced-preview-container');
            }
        });
};

window.runTransformation = function (datasetId) {
    const column = document.getElementById('transform-col').value;
    const method = document.getElementById('transform-method').value;

    document.getElementById('advanced-preview-container').innerHTML = 'Processing...';

    fetch(`/api/data/${datasetId}/clean/advanced`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: 'transform', column, method })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('advanced-preview-container').innerHTML = `<p class="error">${data.error}</p>`;
            } else {
                renderTable(data.preview, 'advanced-preview-container');
            }
        });
};

window.runFeatureEngineering = function (datasetId) {
    const newColName = document.getElementById('new-col-name').value;
    const expression = document.getElementById('feature-expr').value;

    if (!newColName || !expression) {
        alert('Please provide both column name and expression');
        return;
    }

    document.getElementById('advanced-preview-container').innerHTML = 'Processing...';

    fetch(`/api/data/${datasetId}/feature_engineering`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ new_col_name: newColName, expression })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('advanced-preview-container').innerHTML = `<p class="error">${data.error}</p>`;
            } else {
                renderTable(data.preview, 'advanced-preview-container');
            }
        });
};

// Visualization
window.runCorrelationHeatmap = function (datasetId) {
    const container = document.getElementById('heatmap-container');
    container.innerHTML = 'Generating heatmap...';

    fetch(`/api/data/${datasetId}/visualize/correlation`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = `<p class="error">${data.error}</p>`;
                return;
            }

            // Create a simple heatmap table
            let html = '<table style="border-collapse: collapse;">';
            html += '<tr><th></th>';
            data.columns.forEach(col => html += `<th style="padding: 5px; background: var(--bg-light); font-size: 0.85rem;">${col}</th>`);
            html += '</tr>';

            data.matrix.forEach((row, i) => {
                html += `<tr><th style="padding: 5px; background: var(--bg-light); font-size: 0.85rem;">${data.columns[i]}</th>`;
                row.forEach(val => {
                    const intensity = Math.abs(val);
                    const color = val > 0 ? `rgba(99, 102, 241, ${intensity})` : `rgba(239, 68, 68, ${intensity})`;
                    html += `<td style="padding: 10px; background: ${color}; text-align: center; color: ${intensity > 0.5 ? 'white' : 'black'}; font-size: 0.8rem;">${val.toFixed(2)}</td>`;
                });
                html += '</tr>';
            });
            html += '</table>';

            container.innerHTML = html;
        });
};

window.runBoxPlot = function (datasetId) {
    const column = document.getElementById('boxplot-col').value;

    if (!column) {
        alert('Please select a column');
        return;
    }

    fetch(`/api/data/${datasetId}/visualize/boxplot?column=${column}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            const ctx = document.getElementById('boxplot-canvas').getContext('2d');

            // Simple box plot using Chart.js (box-and-whisker)
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [column],
                    datasets: [{
                        label: 'Min',
                        data: [data.min],
                        backgroundColor: 'rgba(99, 102, 241, 0.3)'
                    }, {
                        label: 'Q1',
                        data: [data.q1],
                        backgroundColor: 'rgba(99, 102, 241, 0.5)'
                    }, {
                        label: 'Median',
                        data: [data.median],
                        backgroundColor: 'rgba(99, 102, 241, 0.7)'
                    }, {
                        label: 'Q3',
                        data: [data.q3],
                        backgroundColor: 'rgba(99, 102, 241, 0.5)'
                    }, {
                        label: 'Max',
                        data: [data.max],
                        backgroundColor: 'rgba(99, 102, 241, 0.3)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        });
};

// Reviews
document.addEventListener('DOMContentLoaded', function () {
    const reviewForm = document.getElementById('review-form');
    if (reviewForm) {
        reviewForm.addEventListener('submit', function (e) {
            e.preventDefault();

            const rating = document.getElementById('review-rating').value;
            const comment = document.getElementById('review-comment').value;

            if (!rating) {
                alert('Please select a rating');
                return;
            }

            fetch(`/marketplace/dataset/${DATASET_ID}/review`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ rating: parseInt(rating), comment })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Review submitted successfully!');
                        location.reload();
                    } else {
                        alert(data.error || 'Error submitting review');
                    }
                })
                .catch(err => alert('Error submitting review'));
        });
    }
});


// ====================
// SQL Query Functions
// ====================
function executeSQL(datasetId) {
    const query = document.getElementById('sql-query-input').value.trim();
    if (!query) {
        alert('Please enter a SQL query');
        return;
    }
    
    const statusEl = document.getElementById('sql-status');
    statusEl.textContent = 'Executing...';
    statusEl.style.color = 'var(--primary-color)';
    
    fetch(`/api/data/${datasetId}/query`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusEl.textContent = `✓ Query executed successfully (${data.row_count} rows returned)`;
            statusEl.style.color = '#10b981';
            renderTable(data.results, 'sql-results-container');
        } else {
            statusEl.textContent = `✗ Error: ${data.error}`;
            statusEl.style.color = '#ef4444';
            document.getElementById('sql-results-container').innerHTML = `<div class="error-message" style="padding: 1rem; background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; border-radius: 4px; color: #ef4444;">${data.error}</div>`;
        }
    })
    .catch(err => {
        statusEl.textContent = '✗ Network error';
        statusEl.style.color = '#ef4444';
    });
}

function insertSampleQuery() {
    const  samples = [
        'SELECT * FROM dataset LIMIT 10',
        'SELECT COUNT(*) as total_rows FROM dataset',
        'SELECT * FROM dataset WHERE <column> > 100',
        'SELECT <column>, COUNT(*) as count FROM dataset GROUP BY <column>'
    ];
    const query = samples[Math.floor(Math.random() * samples.length)];
    document.getElementById('sql-query-input').value = query;
}

// ====================
// Quality Score Functions
// ====================
function loadQualityScore(datasetId) {
    const container = document.getElementById('quality-report-container');
    container.innerHTML = '<p style="color: var(--text-secondary);"><i class="fas fa-spinner fa-spin"></i> Calculating quality score...</p>';
    
    fetch(`/api/data/${datasetId}/quality`)
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            container.innerHTML = `<p class="error">${data.error}</p>`;
            return;
        }
        
        const grade = data.grade;
        const gradeColors = {
            'A': '#10b981',
            'B': '#3b82f6',
            'C': '#f59e0b',
            'D': '#f97316',
            'F': '#ef4444'
        };
        
        container.innerHTML = `
            <div class="quality-report" style="background: var(--bg-card); padding: 2rem; border-radius: 12px; border: 1px solid var(--border);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                    <div>
                        <h2 style="margin: 0; font-size: 3rem; color: ${gradeColors[grade]};">${grade}</h2>
                        <p style="margin: 0; color: var(--text-secondary);">Overall Quality Grade</p>
                    </div>
                    <div style="text-align: right;">
                        <h3 style="margin: 0; font-size: 2rem; color: var(--text-primary);">${data.overall_score}%</h3>
                        <p style="margin: 0; color: var(--text-secondary);">Quality Score</p>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
                    <div class="metric-card" style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 8px;">
                        <h4 style="margin: 0 0 0.5rem 0; color: var(--text-secondary); font-size: 0.9rem;">Completeness</h4>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">${data.metrics.completeness}%</div>
                        <div style="background: var(--bg-primary); height: 8px; border-radius: 4px; margin-top: 0.5rem; overflow: hidden;">
                            <div style="background: #10b981; height: 100%; width: ${data.metrics.completeness}%;"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card" style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 8px;">
                        <h4 style="margin: 0 0 0.5rem 0; color: var(--text-secondary); font-size: 0.9rem;">Uniqueness</h4>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">${data.metrics.uniqueness}%</div>
                        <div style="background: var(--bg-primary); height: 8px; border-radius: 4px; margin-top: 0.5rem; overflow: hidden;">
                            <div style="background: #3b82f6; height: 100%; width: ${data.metrics.uniqueness}%;"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card" style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 8px;">
                        <h4 style="margin: 0 0 0.5rem 0; color: var(--text-secondary); font-size: 0.9rem;">Validity</h4>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">${data.metrics.validity}%</div>
                        <div style="background: var(--bg-primary); height: 8px; border-radius: 4px; margin-top: 0.5rem; overflow: hidden;">
                            <div style="background: #8b5cf6; height: 100%; width: ${data.metrics.validity}%;"></div>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 2rem; padding: 1rem; background: var(--bg-secondary); border-radius: 8px;">
                    <h4 style="margin: 0 0 1rem 0;">Dataset Info</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem;">
                        <div><strong>Rows:</strong> ${data.metrics.total_rows.toLocaleString()}</div>
                        <div><strong>Columns:</strong> ${data.metrics.total_columns}</div>
                    </div>
                </div>
            </div>
        `;
    })
    .catch(err => {
        container.innerHTML = `<p class="error">Network error: ${err.message}</p>`;
    });
}

// ====================
// Sample Download Functions
// ====================
function downloadSample(datasetId, percentage) {
    const btn = event.currentTarget;
    const originalHTML = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    btn.disabled = true;
    
    fetch(`/api/data/${datasetId}/sample`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({percentage})
    })
    .then(response => {
        if (!response.ok) throw new Error('Failed to generate sample');
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sample_${Math.round(percentage * 100)}pct.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        
        btn.innerHTML = '<i class="fas fa-check"></i> Downloaded!';
        setTimeout(() => {
            btn.innerHTML = originalHTML;
            btn.disabled = false;
        }, 2000);
    })
    .catch(err => {
        alert(`Error: ${err.message}`);
        btn.innerHTML = originalHTML;
        btn.disabled = false;
    });
}

// ====================
// Dashboard Functions
// ====================
function loadDashboards(datasetId) {
    const container = document.getElementById('dashboards-container');
    container.innerHTML = '<p style="color: var(--text-secondary);"><i class="fas fa-spinner fa-spin"></i> Generating dashboards...</p>';
    
    fetch(`/api/data/${datasetId}/dashboard`)
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            container.innerHTML = `<p class="error">${data.error}</p>`;
            return;
        }
        
        // Load Plotly if not already loaded
        if (typeof Plotly === 'undefined') {
            const script = document.createElement('script');
            script.src = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
           script.onload = () => renderDashboards(data, container);
            document.head.appendChild(script);
        } else {
            renderDashboards(data, container);
        }
    })
    .catch(err => {
        container.innerHTML = `<p class="error">Error loading dashboards: ${err.message}</p>`;
    });
}

function renderDashboards(dashboardData, container) {
    container.innerHTML = '';
    
    if (!dashboardData.charts || dashboardData.charts.length === 0) {
        container.innerHTML = '<p>No charts available for this dataset</p>';
        return;
    }
    
    dashboardData.charts.forEach((chart, index) => {
        const chartDiv = document.createElement('div');
        chartDiv.id = `chart-${index}`;
        chartDiv.style.cssText = 'background: var(--bg-card); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid var(--border);';
        container.appendChild(chartDiv);
        
        let plotData, layout;
        
        if (chart.type === 'histogram') {
            plotData = [{
                x: chart.data,
                type: 'histogram',
                marker: {color: '#667eea'}
            }];
            layout = {
                title: chart.title,
                xaxis: {title: chart.column},
                yaxis: {title: 'Count'},
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            };
        } else if (chart.type === 'bar') {
            plotData = [{
                x: chart.labels,
                y: chart.values,
                type: 'bar',
                marker: {color: '#f5576c'}
            }];
            layout = {
                title: chart.title,
                xaxis: {title: chart.column},
                yaxis: {title: 'Count'},
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            };
        } else if (chart.type === 'scatter') {
            plotData = [{
                x: chart.x,
                y: chart.y,
                mode: 'markers',
                type: 'scatter',
                marker: {color: '#4facfe', size: 6}
            }];
            layout = {
                title: chart.title,
                xaxis: {title: chart.x_label},
                yaxis: {title: chart.y_label},
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            };
        } else if (chart.type === 'heatmap') {
            plotData = [{
                z: chart.z,
                x: chart.x,
                y: chart.y,
                type: 'heatmap',
                colorscale: 'Viridis'
            }];
            layout = {
                title: chart.title,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            };
        }
        
        Plotly.newPlot(chartDiv.id, plotData, layout, {responsive: true});
    });
}

// ====================
// Profile Report Functions
// ====================
function loadProfile(datasetId) {
    const container = document.getElementById('profile-container');
    container.innerHTML = '<p style="color: var(--text-secondary);"><i class="fas fa-spinner fa-spin"></i> Generating profile report...</p>';
    
    fetch(`/api/data/${datasetId}/profile`)
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            container.innerHTML = `<p class="error">${data.error}</p>`;
            return;
        }
        
        renderProfile(data, container);
    })
    .catch(err => {
        container.innerHTML = `<p class="error">Error loading profile: ${err.message}</p>`;
    });
}

function renderProfile(profile, container) {
    let html = `
        <div class="profile-report" style="background: var(--bg-card); padding: 2rem; border-radius: 12px; border: 1px solid var(--border);">
            <h3>Dataset Overview</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
                <div class="stat-box" style="background: var(--bg-secondary); padding: 1rem; border-radius: 8px;">
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">Total Rows</div>
                    <div style="font-size: 1.75rem; font-weight: 700; color: var(--text-primary);">${profile.dataset_info.total_rows.toLocaleString()}</div>
                </div>
                <div class="stat-box" style="background: var(--bg-secondary); padding: 1rem; border-radius: 8px;">
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">Total Columns</div>
                    <div style="font-size: 1.75rem; font-weight: 700; color: var(--text-primary);">${profile.dataset_info.total_columns}</div>
                </div>
                <div class="stat-box" style="background: var(--bg-secondary); padding: 1rem; border-radius: 8px;">
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">Duplicate Rows</div>
                    <div style="font-size: 1.75rem; font-weight: 700; color: var(--text-primary);">${profile.dataset_info.duplicate_rows.toLocaleString()}</div>
                </div>
            </div>
            
            <h3 style="margin-top: 2rem;">Column Analysis</h3>
            <div class="columns-accordion">
    `;
    
    for (const [colName, colData] of Object.entries(profile.columns)) {
        html += `
            <details style="background: var(--bg-secondary); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <summary style="cursor: pointer; font-weight: 600; color: var(--text-primary);">
                    ${colName} (${colData.type})
                    ${colData.missing_percentage > 0 ? `<span style="color: #f59e0b; margin-left: 1rem;">⚠ ${colData.missing_percentage.toFixed(1)}% missing</span>` : ''}
                </summary>
                <div style="margin-top: 1rem; padding-left: 1rem;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem;">
                        <div><strong>Missing:</strong> ${colData.missing_count} (${colData.missing_percentage.toFixed(2)}%)</div>
                        <div><strong>Unique:</strong> ${colData.unique_count}</div>
        `;
        
        if (colData.min !== undefined) {
            html += `
                        <div><strong>Min:</strong> ${colData.min?.toFixed(2) ?? 'N/A'}</div>
                        <div><strong>Max:</strong> ${colData.max?.toFixed(2) ?? 'N/A'}</div>
                        <div><strong>Mean:</strong> ${colData.mean?.toFixed(2) ?? 'N/A'}</div>
                        <div><strong>Std Dev:</strong> ${colData.stddev?.toFixed(2) ?? 'N/A'}</div>
                        <div><strong>Zeros:</strong> ${colData.zeros_count}</div>
                        <div><strong>Negatives:</strong> ${colData.negatives_count}</div>
            `;
            
            if (colData.outliers_count !== undefined) {
                html += `
                        <div><strong>Outliers:</strong> ${colData.outliers_count} (${colData.outliers_percentage.toFixed(2)}%)</div>
                `;
            }
        }
        
        if (colData.top_values) {
            html += `
                    </div>
                    <div style="margin-top: 1rem;">
                        <strong>Top Values:</strong>
                        <ul style="margin-top: 0.5rem;">
            `;
            colData.top_values.slice(0, 5).forEach(item => {
                html += `<li>${item.value}: ${item.count} (${item.percentage.toFixed(1)}%)</li>`;
            });
            html += `
                        </ul>
            `;
        } else {
            html += `</div>`;
        }
        
        html += `
                </div>
            </details>
        `;
    }
    
    html += `
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ====================
// Natural Language Query Functions
// ====================
function convertNLToSQL(datasetId) {
    const nlQuery = document.getElementById('nl-query-input').value.trim();
    if (!nlQuery) {
        alert('Please enter a natural language query');
        return;
    }
    
    const statusEl = document.getElementById('nl-status');
    statusEl.textContent = 'Converting to SQL...';
    statusEl.style.color = 'var(--primary-color)';
    
    fetch(`/api/data/${datasetId}/nl-query`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: nlQuery})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Insert generated SQL into SQL input
            document.getElementById('sql-query-input').value = data.sql;
            statusEl.textContent = '✓ Converted to SQL';
            statusEl.style.color = '#10b981';
        } else {
            statusEl.textContent = `✗ Error: ${data.error}`;
            statusEl.style.color = '#ef4444';
        }
    })
    .catch(err => {
        statusEl.textContent = '✗ Network error';
        statusEl.style.color = '#ef4444';
    });
}
