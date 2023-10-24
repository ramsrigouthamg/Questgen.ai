import React from 'react';
import ReactDOM from 'react-dom/client';
import './assets/css/index.css';
import Home from './page/home';
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <Home />
  </React.StrictMode>
);

reportWebVitals();
