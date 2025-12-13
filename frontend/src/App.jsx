import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import 'flag-icons/css/flag-icons.min.css'
const API_URL = 'http://localhost:8000'

function App() {
  const [activeSection, setActiveSection] = useState('business')
  
  // BO3 state
  const [formData, setFormData] = useState({
    ingredients: '',
  })
  const [prediction, setPrediction] = useState(null)
  const [loadingBO3, setLoadingBO3] = useState(false)
  
  // BO1 state
  const [recipeName, setRecipeName] = useState('')
  const [recommendations, setRecommendations] = useState(null)
  const [loadingBO1, setLoadingBO1] = useState(false)
  
  // BO5 state
  const [ingredientsBO5, setIngredientsBO5] = useState('')
  const [cuisineResult, setCuisineResult] = useState(null)
  const [loadingBO5, setLoadingBO5] = useState(false)
  
  // Shared state
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState(null)

  useEffect(() => {
    // Check API status on mount
    axios.get(`${API_URL}/`)
      .then(res => setApiStatus(res.data))
      .catch(() => setApiStatus({ status: 'offline' }))
  }, [])

  const handleInputChange = (e) => {
    const { name, value, type } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoadingBO3(true)
    setError(null)
    setPrediction(null)

    try {
      const ingredientsList = formData.ingredients
        .split(',')
        .map(ing => ing.trim().toLowerCase())
        .filter(ing => ing.length > 0)

      const payload = {
        ...formData,
        ingredients: ingredientsList
      }

      const response = await axios.post(`${API_URL}/bo3/predict`, payload)
      setPrediction(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction. Make sure the API is running.')
    } finally {
      setLoadingBO3(false)
    }
  }

  const handleRecommendSearch = async (e) => {
    e.preventDefault()
    setLoadingBO1(true)
    setError(null)
    setRecommendations(null)

    try {
      const response = await axios.post(`${API_URL}/bo1/recommend`, {
        recipe_name: recipeName,
        n_recommendations: 5
      })
      setRecommendations(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get recommendations. Make sure the API is running.')
    } finally {
      setLoadingBO1(false)
    }
  }

  const handleCuisineClassification = async (e) => {
    e.preventDefault()
    setLoadingBO5(true)
    setError(null)
    setCuisineResult(null)

    try {
      const ingredientsList = ingredientsBO5
        .split(',')
        .map(ing => ing.trim().toLowerCase())
        .filter(ing => ing.length > 0)

      if (ingredientsList.length === 0) {
        setError('Please enter at least one ingredient')
        setLoadingBO5(false)
        return
      }

      const response = await axios.post(`${API_URL}/bo5/classify`, {
        ingredients: ingredientsList
      })
      setCuisineResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to classify cuisine. Make sure the API is running.')
    } finally {
      setLoadingBO5(false)
    }
  }

  // Helper function to get flag emoji for cuisine
  const getCuisineFlag = (cuisine) => {
    const flags = {
      'Italian': 'it',
      'Mexican': 'mx',
      'Asian': 'cn',  // Using China flag for Asian
      'Indian': 'in',
      'French': 'fr',
      'American': 'us'
    }
    return flags[cuisine] || 'un'  // UN flag as default
  }

  const renderPlaceholder = (boNumber, title) => (
    <div className="placeholder-section">
      <div className="placeholder-content">
        <h2>Business Objective {boNumber}</h2>
        <h3>{title}</h3>
        <p>This feature is currently under development and will be available soon.</p>
      </div>
    </div>
  )

  return (
    <div className="App">
      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-brand">
          <span className="brand-text">Hexateam</span>
        </div>
        <div className="nav-links">
          <button
            className={`nav-link ${activeSection === 'business' ? 'active' : ''}`}
            onClick={() => setActiveSection('business')}
          >
            Business Understanding
          </button>
          {['bo1', 'bo3', 'bo5'].map(bo => (
            <button
              key={bo}
              className={`nav-link ${activeSection === bo ? 'active' : ''}`}
              onClick={() => setActiveSection(bo)}
            >
              {bo.toUpperCase()}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <div className="main-content">
        {activeSection === 'business' && (
          <div className="business-section">
            <div className="trapezoid-grid">
              {/* BO1 */}
              <div className="trapezoid-card left">
                <div className="trapezoid-content">
                  <div className="bo-badge">BO1</div>
                  <h3>Recommandation de Recettes Personnalisées</h3>
                  <div className="dso-section">
                    <strong>DSO 1:</strong>
                    <p>Système de recommandation intelligent basé sur les valeurs nutritionnelles, ingrédients et catégories utilisant K-Nearest Neighbors</p>
                  </div>
                  <button className="goto-btn" onClick={() => setActiveSection('bo1')}>Accéder à BO1 →</button>
                </div>
              </div>

              {/* BO3 */}
              <div className="trapezoid-card right">
                <div className="trapezoid-content">
                  <div className="bo-badge">BO3</div>
                  <h3>Prédiction de Notes de Recettes</h3>
                  <div className="dso-section">
                    <strong>DSO 3:</strong>
                    <p>Modèle de régression Random Forest pour prédire la note d'une recette à partir des ingrédients uniquement</p>
                  </div>
                  <button className="goto-btn" onClick={() => setActiveSection('bo3')}>Accéder à BO3 →</button>
                </div>
              </div>

              {/* BO5 */}
              <div className="trapezoid-card left">
                <div className="trapezoid-content">
                  <div className="bo-badge">BO5</div>
                  <h3>Classification de Type de Cuisine</h3>
                  <div className="dso-section">
                    <strong>DSO 5:</strong>
                    <p>Classification supervisée pour identifier le type de cuisine (Italien, Mexicain, Asiatique, etc.) à partir des ingrédients</p>
                  </div>
                  <button className="goto-btn" onClick={() => setActiveSection('bo5')}>Accéder à BO5 →</button>
                </div>
              </div>
            </div>
          </div>
        )}
        {activeSection === 'bo1' && (
          <div className="bo3-section">
            <div className="section-header">
              <h1>Système de Recommandation de Recettes</h1>
              <p>Trouvez des recettes nutritionnellement similaires avec l'algorithme K-Nearest Neighbors</p>
            </div>

            <div className="dual-container">
              {/* Left: Input */}
              <div className="input-container">
                <h2>Rechercher une Recette</h2>
                <form onSubmit={handleRecommendSearch}>
                  <input 
                    type="text"
                    value={recipeName} 
                    onChange={(e) => setRecipeName(e.target.value)}
                    placeholder="Entrez le nom d'une recette (ex: 'glace', 'gâteau au chocolat')"
                    className="ingredients-textarea"
                    style={{ height: '60px', resize: 'none' }}
                    required
                  />
                  <button type="submit" className="predict-btn" disabled={loadingBO1}>
                    {loadingBO1 ? (
                      <>
                        <span className="spinner"></span>
                        Recherche en cours...
                      </>
                    ) : (
                      'Trouver des Recettes Similaires'
                    )}
                  </button>
                  <div style={{ marginTop: '20px', fontSize: '0.9rem', color: '#64748b' }}>
                    <strong>Comment ça marche:</strong>
                    <ul style={{ marginTop: '10px', paddingLeft: '20px' }}>
                      <li>Entrez un nom de recette pour rechercher</li>
                      <li>KNN trouve les 5 recettes les plus similaires nutritionnellement</li>
                      <li>Similarité basée sur 9 caractéristiques nutritionnelles</li>
                      <li>Résultats avec distance & pourcentage de similarité</li>
                    </ul>
                  </div>
                </form>
              </div>

              {/* Right: Source Recipe */}
              <div className="results-container">
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  marginBottom: '20px'
                }}>
                  <h2 style={{ margin: 0 }}>Résultats de Recherche</h2>
                  {recommendations && (
                    <span style={{
                      padding: '6px 12px',
                      background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                      color: 'white',
                      borderRadius: '20px',
                      fontSize: '0.75rem',
                      fontWeight: '600',
                      letterSpacing: '0.5px'
                    }}>
                      PLUS SIMILAIRE
                    </span>
                  )}
                </div>
                
                {!recommendations && !error && (
                  <div className="empty-state">
                    <p>Entrez un nom de recette pour trouver des alternatives similaires</p>
                  </div>
                )}

                {error && (
                  <div className="error-display">
                    <h3>Erreur de Recherche</h3>
                    <p>{error}</p>
                  </div>
                )}

                {recommendations && (
                  <div className="prediction-display">
                    <div className="rating-section" style={{ background: 'linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)' }}>
                      <div className="rating-label" style={{ fontSize: '0.9rem', marginBottom: '10px' }}>RECETTE SOURCE</div>
                      <div className="rating-value" style={{ fontSize: '1.8rem', marginBottom: '15px' }}>
                        {recommendations.source_recipe.name}
                      </div>
                    </div>

                    {/* Source Recipe Nutrition */}
                    <div style={{ marginTop: '20px', padding: '15px', background: '#f8fafc', borderRadius: '8px' }}>
                      <strong style={{ fontSize: '0.9rem', color: '#475569' }}>Nutrition de la Source:</strong>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginTop: '10px', fontSize: '0.85rem' }}>
                        {Object.entries(recommendations.source_recipe.nutrition).map(([key, value]) => (
                          <div key={key} style={{ padding: '8px', background: 'white', borderRadius: '6px' }}>
                            <div style={{ color: '#64748b', marginBottom: '4px' }}>{key}</div>
                            <div style={{ color: '#10b981', fontWeight: '600' }}>{value}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Similar Recipes - Full Width Below */}
            {recommendations && recommendations.recommendations && recommendations.recommendations.length > 0 && (
              <div className="similar-section">
                <h2>Recettes Similaires Recommandées</h2>
                <div className="recipes-grid">
                  {recommendations.recommendations.map((recipe, idx) => (
                    <div key={idx} className="recipe-card-modern">
                      <div className="recipe-header">
                        <h3>{recipe.name}</h3>
                        <div className="recipe-rating-badge" style={{ background: '#d1fae5' }}>
                          <span className="badge-text" style={{ color: '#065f46', fontSize: '1rem' }}>
                            {recipe.similarity_percentage}%
                          </span>
                        </div>
                      </div>

                      <div className="recipe-body">
                        <div className="ingredients-section">
                          <strong>Nutritional Profile:</strong>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginTop: '10px' }}>
                            {Object.entries(recipe.nutrition).map(([key, value]) => (
                              <div key={key} style={{ padding: '6px', background: '#f1f5f9', borderRadius: '4px', fontSize: '0.75rem' }}>
                                <div style={{ color: '#64748b', marginBottom: '2px' }}>{key}</div>
                                <div style={{ color: '#10b981', fontWeight: '600' }}>{value}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeSection === 'bo2' && renderPlaceholder(2, 'Recipe Classification')}
        {activeSection === 'bo4' && renderPlaceholder(4, 'Recipe Recommendation System')}
        
        {activeSection === 'bo5' && (
          <div className="bo3-section">
            <div className="section-header">
              <h1>Classificateur de Type de Cuisine</h1>
              <p>Identifiez le type de cuisine à partir des ingrédients en utilisant le modèle XGBoost</p>
            </div>

            <div className="dual-container">
              {/* Left: Input */}
              <div className="input-container">
                <h2>Entrez les Ingrédients</h2>
                <form onSubmit={handleCuisineClassification}>
                  <textarea 
                    value={ingredientsBO5} 
                    onChange={(e) => setIngredientsBO5(e.target.value)}
                    placeholder="Entrez les ingrédients séparés par des virgules&#10;&#10;Exemple: tomate, basilic, mozzarella, parmesan, huile d'olive, ail"
                    rows="6"
                    className="ingredients-textarea"
                    required
                  />
                  <button type="submit" className="predict-btn" disabled={loadingBO5}>
                    {loadingBO5 ? (
                      <>
                        <span className="spinner"></span>
                        Classification...
                      </>
                    ) : (
                      'Classifier la Cuisine'
                    )}
                  </button>
                </form>
                
                <div style={{ marginTop: '20px', fontSize: '0.9rem', color: '#64748b', lineHeight: '1.6' }}>
                  <strong style={{ color: '#10b981' }}>Types de Cuisine:</strong>
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(2, 1fr)', 
                    gap: '8px', 
                    marginTop: '10px',
                    fontSize: '0.85rem'
                  }}>
                    <div style={{ padding: '8px', background: '#f8fafc', borderRadius: '6px', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                      <span className="fi fi-it"></span> Italien
                    </div>
                    <div style={{ padding: '8px', background: '#f8fafc', borderRadius: '6px', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                      <span className="fi fi-mx"></span> Mexicain
                    </div>
                    <div style={{ padding: '8px', background: '#f8fafc', borderRadius: '6px', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                      <span className="fi fi-cn"></span> Asiatique
                    </div>
                    <div style={{ padding: '8px', background: '#f8fafc', borderRadius: '6px', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                      <span className="fi fi-in"></span> Indien
                    </div>
                    <div style={{ padding: '8px', background: '#f8fafc', borderRadius: '6px', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                      <span className="fi fi-fr"></span> Français
                    </div>
                    <div style={{ padding: '8px', background: '#f8fafc', borderRadius: '6px', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                      <span className="fi fi-us"></span> Américain
                    </div>
                  </div>
                  <ul style={{ marginTop: '12px', paddingLeft: '20px' }}>
                    <li>Classificateur XGBoost entraîné sur 6 types de cuisine</li>
                    <li>Distribution de probabilité pour toutes les cuisines</li>
                    <li>Score de confiance pour la prédiction</li>
                  </ul>
                </div>
              </div>

              {/* Right: Results */}
              <div className="results-container">
                <h2>Résultats de Classification</h2>
                
                {!cuisineResult && !error && (
                  <div className="empty-state">
                    <p>Entrez des ingrédients pour identifier le type de cuisine</p>
                  </div>
                )}

                {error && (
                  <div className="error-display">
                    <h3>Erreur de Classification</h3>
                    <p>{error}</p>
                  </div>
                )}

                {cuisineResult && (
                  <div className="prediction-display">
                    <div 
                      className="rating-section" 
                      style={{
                        background: 'radial-gradient(circle at 30% 40%, #ecfdf5 0%, #d1fae5 50%, #a7f3d0 100%)'
                      }}
                    >
                      <div className="rating-label" style={{ fontSize: '0.9rem', marginBottom: '10px' }}>CUISINE PRÉDITE</div>
                      <div style={{ fontSize: '5rem', marginBottom: '15px' }}>
                        <span className={`fi fi-${getCuisineFlag(cuisineResult.predicted_cuisine)}`} 
                              style={{ fontSize: '5rem', borderRadius: '8px' }}></span>
                      </div>
                      <div className="rating-value" style={{ fontSize: '2rem' }}>{cuisineResult.predicted_cuisine}</div>
                      <div className="rating-label" style={{ fontSize: '1rem', marginTop: '10px' }}>
                        {cuisineResult.confidence}% Confiance
                      </div>
                    </div>

                    {/* Ingredients Info */}
                    <div style={{ marginTop: '20px', padding: '15px', background: '#f8fafc', borderRadius: '8px' }}>
                      <strong style={{ fontSize: '0.9rem', color: '#475569' }}>Analyse des Ingrédients:</strong>
                      <div style={{ marginTop: '10px', fontSize: '0.85rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                          <span style={{ color: '#64748b' }}>Total Ingrédients:</span>
                          <span style={{ color: '#10b981', fontWeight: '600' }}>{cuisineResult.ingredients.length}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: '#64748b' }}>Caractéristiques Correspondantes:</span>
                          <span style={{ color: '#10b981', fontWeight: '600' }}>{cuisineResult.matched_ingredients.length}</span>
                        </div>
                      </div>
                    </div>

                    {/* Top 3 Predictions */}
                    {cuisineResult.top_3_predictions && cuisineResult.top_3_predictions.length > 0 && (
                      <div style={{ marginTop: '20px' }}>
                        <strong style={{ fontSize: '0.9rem', color: '#475569' }}>Top Prédictions:</strong>
                        <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                          {cuisineResult.top_3_predictions.map((pred, idx) => (
                            <div 
                              key={idx}
                              style={{ 
                                padding: '12px', 
                                background: idx === 0 ? '#d1fae5' : '#f1f5f9', 
                                borderRadius: '8px',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                              }}
                            >
                              <span style={{ 
                                fontWeight: idx === 0 ? '700' : '600',
                                color: idx === 0 ? '#065f46' : '#475569',
                                fontSize: '0.9rem',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px'
                              }}>
                                <span>{idx + 1}.</span>
                                <span className={`fi fi-${getCuisineFlag(pred.cuisine)}`} style={{ fontSize: '1.5rem' }}></span>
                                <span>{pred.cuisine}</span>
                              </span>
                              <span style={{ 
                                fontWeight: '700',
                                color: idx === 0 ? '#059669' : '#10b981',
                                fontSize: '0.95rem'
                              }}>
                                {pred.probability}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* All Probabilities - Full Width Below */}
            {cuisineResult && cuisineResult.all_probabilities && (
              <div className="similar-section">
                <h2>Distribution de Probabilité</h2>
                <div style={{ 
                  display: 'grid', 
                  gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
                  gap: '20px',
                  marginTop: '20px'
                }}>
                  {Object.entries(cuisineResult.all_probabilities).map(([cuisine, probability]) => (
                    <div 
                      key={cuisine}
                      style={{
                        padding: '20px',
                        background: 'white',
                        borderRadius: '12px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        border: cuisine === cuisineResult.predicted_cuisine ? '2px solid #10b981' : '2px solid transparent'
                      }}
                    >
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'center',
                        marginBottom: '12px'
                      }}>
                        <span style={{ 
                          fontSize: '1.1rem', 
                          fontWeight: '600',
                          color: '#1e293b',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '10px'
                        }}>
                          <span className={`fi fi-${getCuisineFlag(cuisine)}`} style={{ fontSize: '2rem' }}></span>
                          {cuisine}
                        </span>
                        <span style={{ 
                          fontSize: '1.2rem', 
                          fontWeight: '700',
                          color: '#10b981'
                        }}>
                          {probability}%
                        </span>
                      </div>
                      <div style={{ 
                        width: '100%', 
                        height: '12px', 
                        background: '#e2e8f0', 
                        borderRadius: '6px',
                        overflow: 'hidden'
                      }}>
                        <div style={{ 
                          width: `${probability}%`, 
                          height: '100%', 
                          background: cuisine === cuisineResult.predicted_cuisine 
                            ? 'linear-gradient(90deg, #10b981 0%, #059669 100%)'
                            : 'linear-gradient(90deg, #94a3b8 0%, #64748b 100%)',
                          transition: 'width 0.5s ease-out'
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        
        {activeSection === 'bo3' && (
          <div className="bo3-section">
            <div className="section-header">
              <h1>Prédicteur de Notes de Recettes</h1>
              <p>Entrez vos ingrédients pour prédire la note en utilisant le modèle Random Forest</p>
            </div>

            <div className="dual-container">
              {/* Left: Input */}
              <div className="input-container">
                <h2>Ingrédients de la Recette</h2>
                <form onSubmit={handleSubmit}>
                  <textarea 
                    name="ingredients" 
                    value={formData.ingredients} 
                    onChange={handleInputChange}
                    placeholder="Entrez les ingrédients séparés par des virgules&#10;&#10;Exemple: farine, sucre, œufs, beurre, lait, extrait de vanille, levure chimique, sel"
                    rows="6"
                    className="ingredients-textarea"
                    required
                  />
                  <button type="submit" className="predict-btn" disabled={loadingBO3}>
                    {loadingBO3 ? (
                      <>
                        <span className="spinner"></span>
                        Analyse...
                      </>
                    ) : (
                      'Prédire la Note'
                    )}
                  </button>
                </form>
                
                <div style={{ marginTop: '20px', fontSize: '0.9rem', color: '#64748b', lineHeight: '1.6' }}>
                  <strong style={{ color: '#10b981' }}>Comment ça marche:</strong>
                  <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
                    <li>Entrez une liste d'ingrédients que vous avez</li>
                    <li>Notre modèle Random Forest prédit la note de la recette (0-5 étoiles)</li>
                    <li>Obtenez 6 recettes similaires de notre base de données avec pourcentages de correspondance</li>
                  </ul>
                </div>
              </div>

              {/* Right: Results */}
              <div className="results-container">
                <h2>Résultats de Prédiction</h2>
                
                {!prediction && !error && (
                  <div className="empty-state">
                    <p>Entrez des ingrédients et cliquez sur prédire pour voir les résultats</p>
                  </div>
                )}

                {error && (
                  <div className="error-display">
                    <h3>Erreur de Prédiction</h3>
                    <p>{error}</p>
                  </div>
                )}

                {prediction && (
                  <div className="prediction-display">
                    <div 
                      className="rating-section" 
                      style={{
                        background: 'radial-gradient(circle at 30% 40%, #ecfdf5 0%, #d1fae5 50%, #a7f3d0 100%)'
                      }}
                    >
                      <div className="stars-large">
                        {[...Array(5)].map((_, i) => (
                          <span 
                            key={i} 
                            className={i < Math.round(prediction.predicted_rating_stars) ? 'star filled' : 'star'}
                            style={{
                              fontSize: `${2 + (i * 0.3)}rem`,
                              animation: `starAppear 0.3s ease-out ${i * 0.1}s both`
                            }}
                          >
                            ★
                          </span>
                        ))}
                      </div>
                      <div className="rating-value">{prediction.rating_display}</div>
                      <div className="rating-label">Note Prédite</div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Similar Recipes - Full Width Below */}
            {prediction && prediction.similar_recipes && prediction.similar_recipes.length > 0 && (
              <div className="similar-section">
                <h2>Recettes Similaires de la Base de Données</h2>
                <div className="recipes-grid">
                  {prediction.similar_recipes.map((recipe, idx) => (
                    <div key={idx} className="recipe-card-modern">
                      <div className="recipe-header">
                        <h3>{recipe.name}</h3>
                        <div className="recipe-rating-badge">
                          <span className="badge-stars">
                            {[...Array(5)].map((_, i) => (
                              <span key={i} className={i < Math.round(recipe.rating_stars) ? 'star-small filled' : 'star-small'}>★</span>
                            ))}
                          </span>
                          <span className="badge-text">{recipe.rating_stars.toFixed(2)}</span>
                        </div>
                      </div>

                      <div className="recipe-body">
                        <div className="similarity-badge">
                          {recipe.similarity_score}% Correspondance
                        </div>
                        <div className="ingredients-section">
                          <strong>Ingrédients:</strong>
                          <div className="ingredients-tags">
                            {recipe.common_ingredients && recipe.common_ingredients.map((ing, i) => (
                              <span key={`c-${i}`} className="ingredient-tag common">{ing}</span>
                            ))}
                            {recipe.other_ingredients && recipe.other_ingredients.map((ing, i) => (
                              <span key={`o-${i}`} className="ingredient-tag">{ing}</span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
