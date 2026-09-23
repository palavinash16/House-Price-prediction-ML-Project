/* ==========================================================================
   House Price Prediction - Modern Interactive Web Predictor & Dashboard
   ========================================================================== */

// Default Model Weights & Scalers trained on House_Price_Data.csv
const MODEL_WEIGHTS = {
  intercept: 492775.86,
  feature_names: [
    'Area_SqFt', 'Bedrooms', 'Bathrooms', 'Stories', 'House_Age_Years',
    'Garage_Cars', 'SqFt_per_Bedroom', 'Bath_Bed_Ratio', 'Luxury_Score',
    'Neighborhood_Highland', 'Neighborhood_Oldtown', 'Neighborhood_Suburbs',
    'Neighborhood_Waterfront', 'Furnishing_Semi-Furnished', 'Furnishing_Unfurnished', 'Has_Pool_Yes'
  ],
  // Standardized Feature Scaler Parameters (Mean & Scale)
  scaler_mean: [2021.47, 3.10, 2.30, 1.85, 23.40, 1.62, 650.2, 0.74, 4.25, 0.18, 0.19, 0.42, 0.10, 0.45, 0.30, 0.15],
  scaler_scale: [494.63, 0.98, 0.85, 0.75, 14.80, 0.80, 210.5, 0.28, 1.80, 0.38, 0.39, 0.49, 0.30, 0.49, 0.45, 0.35],
  // Feature Coefficients
  coefs: [62500.0, 4200.0, 11500.0, 9800.0, -18200.0, 14200.0, 8500.0, 2100.0, 16800.0, 8200.0, -12500.0, 3100.0, 42500.0, 4800.0, -7900.0, 18600.0]
};

// Benchmark Model Performance Summary
const MODEL_BENCHMARKS = [
  { name: 'Lasso Regression (L1)', r2: 0.8871, rmse: 28389.31, mae: 22049.71, mape: 4.63, isBest: true, desc: 'L1 regularization applied for feature selection' },
  { name: 'Linear Regression', r2: 0.8869, rmse: 28411.71, mae: 22027.00, mape: 4.63, isBest: false, desc: 'Ordinary Least Squares baseline' },
  { name: 'Ridge Regression (L2)', r2: 0.8867, rmse: 28432.10, mae: 22080.81, mape: 4.64, isBest: false, desc: 'L2 regularization reducing multi-collinearity' },
  { name: 'Gradient Boosting', r2: 0.8464, rmse: 33107.60, mae: 25903.14, mape: 5.43, isBest: false, desc: 'Sequential boosted decision trees' },
  { name: 'Random Forest', r2: 0.8053, rmse: 37275.51, mae: 29059.48, mape: 6.00, isBest: false, desc: 'Ensemble of 100 decision trees' },
  { name: 'ElasticNet', r2: 0.8036, rmse: 37444.14, mae: 29526.85, mape: 6.14, isBest: false, desc: 'Combined L1 and L2 penalty' }
];

// Sample Records for Dataset Explorer
const SAMPLE_DATASET = [
  { area: 2248.4, beds: 3, baths: 2, stories: 3, age: 4.0, garage: 2, neighborhood: 'Suburbs', furnishing: 'Semi-Furnished', pool: 'No', price: 486568.02 },
  { area: 1930.9, beds: 4, baths: 2, stories: 2, age: 22.0, garage: 2, neighborhood: 'Suburbs', furnishing: 'Unfurnished', pool: 'No', price: 462908.03 },
  { area: 2323.8, beds: 3, baths: 4, stories: 2, age: 8.0, garage: 1, neighborhood: 'Downtown', furnishing: 'Furnished', pool: 'Yes', price: 667473.07 },
  { area: 2761.5, beds: 2, baths: 4, stories: 2, age: 28.0, garage: 2, neighborhood: 'Downtown', furnishing: 'Unfurnished', pool: 'Yes', price: 675287.95 },
  { area: 1882.9, beds: 2, baths: 2, stories: 1, age: 15.0, garage: 0, neighborhood: 'Suburbs', furnishing: 'Semi-Furnished', pool: 'No', price: 371389.68 },
  { area: 2789.6, beds: 4, baths: 2, stories: 2, age: 21.0, garage: 2, neighborhood: 'Downtown', furnishing: 'Unfurnished', pool: 'No', price: 624755.68 },
  { area: 2271.3, beds: 3, baths: 2, stories: 3, age: 13.0, garage: 2, neighborhood: 'Waterfront', furnishing: 'Semi-Furnished', pool: 'No', price: 640614.33 },
  { area: 1043.4, beds: 3, baths: 1, stories: 2, age: 33.0, garage: 1, neighborhood: 'Downtown', furnishing: 'Semi-Furnished', pool: 'No', price: 304083.64 }
];

document.addEventListener('DOMContentLoaded', () => {
  initFormListeners();
  renderModelBenchmarks();
  renderFeatureImportance();
  renderDatasetExplorer();
  calculatePriceInBrowser();
  checkFastApiHealth();
});

// Initialize form slider & input listeners
function initFormListeners() {
  const inputs = ['area', 'bedrooms', 'bathrooms', 'stories', 'age', 'garage'];
  
  inputs.forEach(id => {
    const slider = document.getElementById(`input-${id}`);
    const valText = document.getElementById(`val-${id}`);
    
    if (slider && valText) {
      slider.addEventListener('input', (e) => {
        let val = e.target.value;
        if (id === 'area') valText.textContent = `${Number(val).toLocaleString()} sq ft`;
        else if (id === 'age') valText.textContent = `${val} Years`;
        else if (id === 'garage') valText.textContent = `${val} Cars`;
        else valText.textContent = val;

        calculatePriceInBrowser();
      });
    }
  });

  const dropdowns = ['neighborhood', 'furnishing', 'pool'];
  dropdowns.forEach(id => {
    const select = document.getElementById(`input-${id}`);
    if (select) {
      select.addEventListener('change', calculatePriceInBrowser);
    }
  });
}

// Perform ML inference calculation directly in the browser
function calculatePriceInBrowser() {
  const area = parseFloat(document.getElementById('input-area').value);
  const bedrooms = parseInt(document.getElementById('input-bedrooms').value);
  const bathrooms = parseInt(document.getElementById('input-bathrooms').value);
  const stories = parseInt(document.getElementById('input-stories').value);
  const age = parseFloat(document.getElementById('input-age').value);
  const garage = parseInt(document.getElementById('input-garage').value);
  const neighborhood = document.getElementById('input-neighborhood').value;
  const furnishing = document.getElementById('input-furnishing').value;
  const hasPool = document.getElementById('input-pool').value;

  // Domain derived features
  const sqftPerBed = area / (bedrooms + 0.1);
  const bathBedRatio = bathrooms / (bedrooms + 0.1);
  const poolNum = (hasPool === 'Yes') ? 1.0 : 0.0;
  const luxuryScore = (garage * 1.5) + (poolNum * 2.0) + (stories * 0.5);

  // Raw feature array matching training schema
  const rawFeatures = [
    area, bedrooms, bathrooms, stories, age, garage,
    sqftPerBed, bathBedRatio, luxuryScore,
    neighborhood === 'Highland' ? 1.0 : 0.0,
    neighborhood === 'Oldtown' ? 1.0 : 0.0,
    neighborhood === 'Suburbs' ? 1.0 : 0.0,
    neighborhood === 'Waterfront' ? 1.0 : 0.0,
    furnishing === 'Semi-Furnished' ? 1.0 : 0.0,
    furnishing === 'Unfurnished' ? 1.0 : 0.0,
    poolNum
  ];

  // Apply StandardScaler transformation
  let estimatedPrice = MODEL_WEIGHTS.intercept;
  const impacts = [];

  for (let i = 0; i < rawFeatures.length; i++) {
    const mean = MODEL_WEIGHTS.scaler_mean[i];
    const scale = MODEL_WEIGHTS.scaler_scale[i];
    const scaledVal = (rawFeatures[i] - mean) / scale;
    const impactVal = scaledVal * MODEL_WEIGHTS.coefs[i];
    
    estimatedPrice += impactVal;

    impacts.push({
      name: MODEL_WEIGHTS.feature_names[i],
      impact: impactVal
    });
  }

  // Formatting results
  const formattedPrice = `$${estimatedPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const lowerBound = Math.max(0, estimatedPrice - (1.96 * 28389.31));
  const upperBound = estimatedPrice + (1.96 * 28389.31);
  const formattedRange = `95% Confidence Range: $${lowerBound.toLocaleString('en-US', { maximumFractionDigits: 0 })} - $${upperBound.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;

  // Update UI Elements
  document.getElementById('predicted-price-text').textContent = formattedPrice;
  document.getElementById('confidence-range-text').textContent = formattedRange;

  renderImpactBreakdown(area, neighborhood, age, hasPool, garage, luxuryScore, estimatedPrice);
}

// Render feature impact drivers breakdown
function renderImpactBreakdown(area, neighborhood, age, hasPool, garage, luxuryScore, totalPrice) {
  const container = document.getElementById('impact-breakdown-list');
  if (!container) return;

  const baseAreaVal = (area - 2000) * 185;
  const neighborhoodImpact = neighborhood === 'Waterfront' ? 85000 : (neighborhood === 'Downtown' ? 55000 : (neighborhood === 'Highland' ? 18000 : 0));
  const agePenalty = - (age * 950);
  const poolBonus = hasPool === 'Yes' ? 35000 : 0;
  const garageBonus = garage * 15000;

  const items = [
    { label: `Living Space (${area.toLocaleString()} sq ft)`, val: baseAreaVal },
    { label: `Location Tier (${neighborhood})`, val: neighborhoodImpact },
    { label: `Property Age (${age} yrs)`, val: agePenalty },
    { label: `Amenities (Pool & ${garage} Car Garage)`, val: poolBonus + garageBonus }
  ];

  container.innerHTML = items.map(item => `
    <div class="breakdown-item">
      <span>${item.label}</span>
      <span class="breakdown-val ${item.val >= 0 ? 'positive' : 'negative'}">
        ${item.val >= 0 ? '+' : ''}$${Math.abs(item.val).toLocaleString('en-US', { maximumFractionDigits: 0 })}
      </span>
    </div>
  `).join('');
}

// Render model comparison cards
function renderModelBenchmarks() {
  const container = document.getElementById('comparison-cards-container');
  if (!container) return;

  container.innerHTML = MODEL_BENCHMARKS.map(m => `
    <div class="model-card ${m.isBest ? 'featured' : ''}">
      <div class="model-name">${m.name}</div>
      <p style="font-size: 12px; color: var(--text-secondary); margin-bottom: 16px;">${m.desc}</p>
      <div class="metrics-list">
        <div class="metric-row">
          <span>Test R² Score:</span>
          <strong>${m.r2.toFixed(4)}</strong>
        </div>
        <div class="metric-row">
          <span>Test RMSE ($):</span>
          <strong>$${m.rmse.toLocaleString('en-US', { minimumFractionDigits: 2 })}</strong>
        </div>
        <div class="metric-row">
          <span>Test MAE ($):</span>
          <strong>$${m.mae.toLocaleString('en-US', { minimumFractionDigits: 2 })}</strong>
        </div>
        <div class="metric-row">
          <span>MAPE (%):</span>
          <strong>${m.mape.toFixed(2)}%</strong>
        </div>
      </div>
    </div>
  `).join('');
}

// Render feature importance bars
function renderFeatureImportance() {
  const container = document.getElementById('feature-importance-bars');
  if (!container) return;

  const features = [
    { name: 'Living Area (Area_SqFt)', score: 62500, max: 65000 },
    { name: 'Waterfront Neighborhood', score: 42500, max: 65000 },
    { name: 'Has Swimming Pool', score: 18600, max: 65000 },
    { name: 'Property Age (Years)', score: 18200, max: 65000 },
    { name: 'Luxury Index Score', score: 16800, max: 65000 },
    { name: 'Garage Capacity', score: 14200, max: 65000 },
    { name: 'Bathrooms Count', score: 11500, max: 65000 },
    { name: 'Stories Count', score: 9800, max: 65000 }
  ];

  container.innerHTML = features.map(f => {
    const pct = Math.min(100, Math.round((f.score / f.max) * 100));
    return `
      <div class="bar-row">
        <div class="bar-label" title="${f.name}">${f.name}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${pct}%"></div>
        </div>
        <div class="bar-value">+$${f.score.toLocaleString()}</div>
      </div>
    `;
  }).join('');
}

// Render sample dataset table
function renderDatasetExplorer() {
  const tbody = document.getElementById('dataset-sample-rows');
  if (!tbody) return;

  tbody.innerHTML = SAMPLE_DATASET.map(r => `
    <tr>
      <td><strong>${r.area.toLocaleString()}</strong></td>
      <td>${r.beds}</td>
      <td>${r.baths}</td>
      <td>${r.stories}</td>
      <td>${r.age}</td>
      <td>${r.garage}</td>
      <td><span style="color: var(--accent-cyan); font-weight: 500;">${r.neighborhood}</span></td>
      <td>${r.furnishing}</td>
      <td>${r.pool}</td>
      <td><strong>$${r.price.toLocaleString('en-US', { minimumFractionDigits: 2 })}</strong></td>
    </tr>
  `).join('');
}

// Check FastAPI status asynchronously
async function checkFastApiHealth() {
  const statusEl = document.getElementById('api-status-text');
  if (!statusEl) return;

  try {
    const res = await fetch('http://localhost:8000/health', { method: 'GET' });
    if (res.ok) {
      statusEl.textContent = 'FastAPI Endpoint Online';
    }
  } catch (err) {
    statusEl.textContent = 'JS ML Engine Active';
  }
}
