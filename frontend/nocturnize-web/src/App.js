import React, { useState } from 'react';
import './App.css';

// Layout Components
import Header from './components/layout/Header';
import Navigation from './components/layout/Navigation';
import Footer from './components/layout/Footer';

// Page Components
import HealthCheckPage from './components/pages/HealthCheckPage';
import ModelsPage from './components/pages/ModelsPage';
import PredictionsPage from './components/pages/PredictionsPage';
import SimulationsPage from './components/pages/SimulationsPage';
import ComparePage from './components/pages/ComparePage';

function App() {
  const [activeTab, setActiveTab] = useState('health');

  const renderPage = () => {
    switch (activeTab) {
      case 'health':
        return <HealthCheckPage />;
      case 'models':
        return <ModelsPage />;
      case 'predict':
        return <PredictionsPage />;
      case 'simulate':
        return <SimulationsPage />;
      case 'compare':
        return <ComparePage />;
      default:
        return <HealthCheckPage />;
    }
  };

  return (
    <div className="App">
      <Header />
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        {renderPage()}
      </main>
      <Footer />
    </div>
  );
}

export default App;