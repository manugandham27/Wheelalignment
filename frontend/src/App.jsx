import React, { useState, useEffect, useRef } from 'react'

function App() {
  // Preset profiles
  const [presets, setPresets] = useState({})
  const [selectedPreset, setSelectedPreset] = useState('Custom Manual Entry')
  
  // Tabular inputs
  const [brand, setBrand] = useState('Michelin')
  const [size, setSize] = useState('205/55R16')
  const [expectedLife, setExpectedLife] = useState(50000)
  const [kmDriven, setKmDriven] = useState(20000)
  const [camber, setCamber] = useState(0.0)
  const [roadCondition, setRoadCondition] = useState('Smooth')
  const [weatherCondition, setWeatherCondition] = useState('Dry')
  const [retreaded, setRetreaded] = useState('No')
  
  // Image upload
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const fileInputRef = useRef(null)
  
  // Pipeline response
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  
  // Explainability toggling (gradcam vs integrated gradients)
  const [explainMethod, setExplainMethod] = useState('gradcam')

  // Fetch presets from backend API on mount
  useEffect(() => {
    fetch('http://localhost:8000/api/presets')
      .then(res => {
        if (!res.ok) throw new Error("Could not fetch presets")
        return res.json()
      })
      .then(data => setPresets(data))
      .catch(err => {
        console.warn("Backend presets offline. Falling back to local values.", err)
        // Fallback local preset mapping
        setPresets({
          "New Tire Baseline": { expected_life: 60000, km_driven: 2000, camber: 0.0, road_condition: "Smooth", weather_condition: "Dry", brand: "Michelin", size: "205/55R16", retreaded: "No" },
          "Over-inflated City Commute": { expected_life: 50000, km_driven: 15000, camber: -0.2, road_condition: "Smooth", weather_condition: "Humid", brand: "Continental", size: "225/65R17", retreaded: "No" },
          "Under-inflated Highway Driving": { expected_life: 55000, km_driven: 35000, camber: 0.5, road_condition: "Smooth", weather_condition: "Cold", brand: "Bridgestone", size: "245/40R18", retreaded: "No" },
          "Rough Off-Road Terrain": { expected_life: 40000, km_driven: 18000, camber: -1.5, road_condition: "Off-road", weather_condition: "Rainy", brand: "Goodyear", size: "195/65R15", retreaded: "Yes" }
        })
      })
  }, [])

  // Apply preset profile changes
  const handlePresetChange = (e) => {
    const val = e.target.value
    setSelectedPreset(val)
    if (val !== 'Custom Manual Entry' && presets[val]) {
      const p = presets[val]
      setBrand(p.brand)
      setSize(p.size)
      setExpectedLife(p.expected_life)
      setKmDriven(p.km_driven)
      setCamber(p.camber)
      setRoadCondition(p.road_condition)
      setWeatherCondition(p.weather_condition)
      setRetreaded(p.retreaded)
    }
  }

  // Handle image upload selection
  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImageFile(file)
      setImagePreview(URL.createObjectURL(file))
    }
  }

  const triggerFileInput = () => {
    fileInputRef.current.click()
  }

  // Submit to FastAPI backend pipeline
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!imageFile) {
      setError("Please select/upload a tire tread photograph first.")
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    // Assemble unified form payload
    const formData = new FormData()
    formData.append('image', imageFile)
    
    const sensor_data = {
      vehicle_model: "Sedan",
      fuel_type: "Petrol",
      transmission_type: "Automatic",
      country: "Germany",
      "maximum_power(hp)": 150,
      "maximum_torque(N/m)": 220,
      "maximum_speed(km/h)": 200,
      "vehicle_acceleration(0-100 km/h in seconds)": 8.5,
      "vehicle_mileage(mpg)": 30.0,
      "vehicle_sprung_mass(kg)": 1500,
      "steering_radius(m)": 5.5,
      "axle_type(driven/dead)": "driven",
      tyre_brand: brand,
      tyre_size: size,
      tread_material: "Silica Compound",
      tread_pattern: "Symmetric",
      "tyre_camber_angle(degree)": camber,
      "standard_tread_depth(mm)": 8.0,
      retreaded: retreaded,
      road_condition: roadCondition,
      weather_condition: weatherCondition,
      "expected_tyre_life(km)": expectedLife,
      "kilometers_driven(km)": kmDriven
    }
    
    formData.append('sensor_data', JSON.stringify(sensor_data))

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || "Prediction request failed.")
      }
      
      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message || "Failed to connect to backend server. Ensure FastAPI is running.")
    } finally {
      setLoading(false)
    }
  }

  // Get status class for styling severity
  const getSeverityClass = (wearClass) => {
    if (wearClass === 'New') return 'ok'
    if (wearClass === 'Serviceable') return 'warn'
    return 'danger'
  }

  return (
    <div className="app-container">
      <header>
        <h1>🚗 Decoupled Tire Diagnostics System</h1>
        <p className="subtitle">Production-grade separation of deep representations, time-series IMU LSTM fusion, and modern React client SPA interface.</p>
      </header>

      <div className="dashboard-grid">
        {/* Left Side: Inputs and Uploads */}
        <div>
          <form onSubmit={handleSubmit}>
            <div className="card">
              <h2 className="card-title">📷 Tread Image Upload</h2>
              <div className="upload-zone" onClick={triggerFileInput}>
                <div className="upload-icon">📤</div>
                <p>Click to browse or drop tire photograph here</p>
                <span style={{fontSize: '0.8rem', color: '#64748b'}}>Supports JPG, JPEG, PNG</span>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleImageChange} 
                  accept="image/*" 
                  className="file-input-hidden"
                />
              </div>
              
              {imagePreview && (
                <div className="preview-container">
                  <p style={{fontSize: '0.85rem', fontWeight: 600, color: '#b0b3c0', marginBottom: 5}}>Image Preview:</p>
                  <img src={imagePreview} className="preview-img" alt="Tire preview" />
                </div>
              )}
            </div>

            <div className="card">
              <h2 className="card-title">🛞 Preset Vehicle Profiles</h2>
              <div className="form-group">
                <label>Select Preset Scenario</label>
                <select value={selectedPreset} onChange={handlePresetChange}>
                  <option value="Custom Manual Entry">Custom Manual Entry</option>
                  {Object.keys(presets).map(pName => (
                    <option key={pName} value={pName}>{pName}</option>
                  ))}
                </select>
              </div>

              <h2 className="card-title" style={{marginTop: '25px', fontSize: '1.1rem'}}>🛠️ Adjustable Sensor Readings</h2>
              
              <div className="form-group">
                <label>Tire Brand</label>
                <select value={brand} onChange={e => { setBrand(e.target.value); setSelectedPreset('Custom Manual Entry'); }}>
                  <option value="Michelin">Michelin</option>
                  <option value="Bridgestone">Bridgestone</option>
                  <option value="Continental">Continental</option>
                  <option value="Goodyear">Goodyear</option>
                  <option value="Pirelli">Pirelli</option>
                </select>
              </div>

              <div className="form-group">
                <label>Tire Size Profile</label>
                <select value={size} onChange={e => { setSize(e.target.value); setSelectedPreset('Custom Manual Entry'); }}>
                  <option value="205/55R16">205/55R16</option>
                  <option value="225/65R17">225/65R17</option>
                  <option value="245/40R18">245/40R18</option>
                  <option value="195/65R15">195/65R15</option>
                </select>
              </div>

              <div className="form-group">
                <label>Expected Lifespan (km)</label>
                <div className="range-slider">
                  <input 
                    type="range" 
                    min="30000" 
                    max="80000" 
                    step="5000" 
                    value={expectedLife} 
                    onChange={e => { setExpectedLife(parseInt(e.target.value)); setSelectedPreset('Custom Manual Entry'); }} 
                  />
                  <span className="slider-val">{expectedLife.toLocaleString()}</span>
                </div>
              </div>

              <div className="form-group">
                <label>Current Distance Driven (km)</label>
                <div className="range-slider">
                  <input 
                    type="range" 
                    min="0" 
                    max={expectedLife} 
                    step="1000" 
                    value={kmDriven} 
                    onChange={e => { setKmDriven(parseInt(e.target.value)); setSelectedPreset('Custom Manual Entry'); }} 
                  />
                  <span className="slider-val">{kmDriven.toLocaleString()}</span>
                </div>
              </div>

              <div className="form-group">
                <label>Camber Alignment Angle (deg)</label>
                <div className="range-slider">
                  <input 
                    type="range" 
                    min="-4.0" 
                    max="4.0" 
                    step="0.1" 
                    value={camber} 
                    onChange={e => { setCamber(parseFloat(e.target.value)); setSelectedPreset('Custom Manual Entry'); }} 
                  />
                  <span className="slider-val">{camber > 0 ? `+${camber.toFixed(1)}` : camber.toFixed(1)}°</span>
                </div>
              </div>

              <div className="form-group">
                <label>Typical Road Surface</label>
                <select value={roadCondition} onChange={e => { setRoadCondition(e.target.value); setSelectedPreset('Custom Manual Entry'); }}>
                  <option value="Smooth">Smooth</option>
                  <option value="Rough">Rough</option>
                  <option value="Off-road">Off-road</option>
                </select>
              </div>

              <div className="form-group">
                <label>Dominant Weather</label>
                <select value={weatherCondition} onChange={e => { setWeatherCondition(e.target.value); setSelectedPreset('Custom Manual Entry'); }}>
                  <option value="Dry">Dry</option>
                  <option value="Humid">Humid</option>
                  <option value="Cold">Cold</option>
                  <option value="Rainy">Rainy</option>
                </select>
              </div>

              <div className="form-group" style={{marginBottom: '30px'}}>
                <label>Retreaded Tire Status</label>
                <select value={retreaded} onChange={e => { setRetreaded(e.target.value); setSelectedPreset('Custom Manual Entry'); }}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <button type="submit" className="btn-primary" disabled={loading || !imageFile}>
                {loading ? 'Processing Model Pipeline...' : '🔍 Analyze Tire Health'}
              </button>
            </div>
          </form>
        </div>

        {/* Right Side: Output Results */}
        <div>
          {error && (
            <div className="card" style={{border: '1px solid rgba(239, 68, 68, 0.3)', background: 'rgba(239, 68, 68, 0.05)'}}>
              <p style={{color: '#f87171', margin: 0, fontWeight: 600}}>⚠️ Error: {error}</p>
            </div>
          )}

          {loading && (
            <div className="card" style={{textAlign: 'center', padding: '60px 20px'}}>
              <div className="spinner"></div>
              <p style={{color: '#94a3b8', fontWeight: 500, margin: 0}}>Running image feed through ResNet-18 Wear Classifier, generating Grad-CAM/Integrated Gradients heatmaps, and running multi-output XGBoost predictions with MC Dropout uncertainty...</p>
            </div>
          )}

          {!loading && !results && !error && (
            <div className="card empty-state">
              <p style={{fontSize: '3rem', margin: '0 0 10px 0'}}>🛞</p>
              <p style={{margin: 0}}>Upload a tire photograph and apply sensor parameters, then click <strong>Analyze Tire Health</strong> to run predictions.</p>
            </div>
          )}

          {results && (
            <div>
              <div className="results-grid">
                {/* Severity Card */}
                <div className={`status-card ${getSeverityClass(results.wear_class)}`}>
                  <div className="status-label">Wear Severity</div>
                  <div className="status-value">{results.wear_class}</div>
                  <div className="status-desc">Classification calculated from deep vision features.</div>
                </div>

                {/* Tread Depth Card (with Uncertainty) */}
                <div className="status-card">
                  <div className="status-label">Estimated Tread Depth</div>
                  <div className="status-value">
                    {results.estimated_tread_depth_mm.toFixed(2)}
                    <span style={{fontSize: '1rem', color: '#a1a1aa', marginLeft: '5px'}}>
                      ± {results.estimated_tread_depth_uncertainty_mm.toFixed(2)} mm
                    </span>
                  </div>
                  <div className="status-desc">Calculated via MC Dropout (95% Bayesian Confidence range).</div>
                </div>

                {/* RUL Card (with Uncertainty) */}
                <div className="status-card">
                  <div className="status-label">Remaining Useful Life (RUL)</div>
                  <div className="status-value">
                    {results.predicted_rul_km.toLocaleString()}
                    <span style={{fontSize: '1rem', color: '#a1a1aa', marginLeft: '5px'}}>
                      ± {results.predicted_rul_uncertainty_km.toLocaleString()} km
                    </span>
                  </div>
                  <div className="status-desc">Fused regression output incorporating tabular operational features.</div>
                </div>

                {/* Alignment Card */}
                <div className={`status-card ${results.alignment_flag ? 'danger' : 'ok'}`}>
                  <div className="status-label">Wheel Alignment</div>
                  <div className="status-value" style={{color: results.alignment_flag ? '#ef4444' : '#10b981'}}>
                    {results.alignment_flag ? 'MISALIGNED' : 'ALIGNED'}
                  </div>
                  <div className="status-desc">Symmetry Confidence: {(results.alignment_confidence * 100).toFixed(1)}%</div>
                </div>
              </div>

              {/* Diagnostic Explanations */}
              <div className="alert-info">
                <strong>🔎 Diagnostic Insight:</strong> {results.diagnosis}
              </div>

              {/* Explainability visualization section */}
              <div className="card">
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px'}}>
                  <h3 className="card-title" style={{fontSize: '1.1rem', margin: 0}}>🔍 Explainability Heatmap</h3>
                  
                  {/* Selector for Grad-CAM vs Integrated Gradients */}
                  <div style={{background: '#151821', borderRadius: '6px', padding: '3px', border: '1px solid rgba(255,255,255,0.08)'}}>
                    <button 
                      type="button"
                      onClick={() => setExplainMethod('gradcam')}
                      style={{
                        padding: '6px 12px', borderRadius: '4px', border: 'none', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
                        background: explainMethod === 'gradcam' ? '#3b82f6' : 'transparent',
                        color: explainMethod === 'gradcam' ? '#ffffff' : '#64748b'
                      }}
                    >
                      Grad-CAM (Coarse)
                    </button>
                    <button 
                      type="button"
                      onClick={() => setExplainMethod('ig')}
                      style={{
                        padding: '6px 12px', borderRadius: '4px', border: 'none', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
                        background: explainMethod === 'ig' ? '#3b82f6' : 'transparent',
                        color: explainMethod === 'ig' ? '#ffffff' : '#64748b'
                      }}
                    >
                      Integrated Gradients (Fine)
                    </button>
                  </div>
                </div>

                <p style={{fontSize: '0.85rem', color: '#b0b3c0', marginBottom: '15px'}}>
                  {explainMethod === 'gradcam' 
                    ? 'Grad-CAM displays the activation maps of the final convolutional block of the backbone.'
                    : 'Integrated Gradients calculates exact pixel-level attributions mapping visual gradients to backpropagation.'
                  }
                </p>

                {explainMethod === 'gradcam' && results.explanation_heatmap_url && (
                  <img src={results.explanation_heatmap_url} className="heatmap-img" alt="Grad-CAM overlay" />
                )}

                {explainMethod === 'ig' && results.explanation_ig_url && (
                  <img src={results.explanation_ig_url} className="heatmap-img" alt="Integrated Gradients overlay" />
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
