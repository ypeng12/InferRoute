// frontend/src/config.ts
// Automatically resolves backend API URL for local dev, Docker container, or deployed HF Space

let defaultApiBase = 'https://inferroute-977n.onrender.com';

if (typeof window !== 'undefined') {
  if (window.location.port === '5173' || window.location.port === '3000' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    defaultApiBase = 'http://127.0.0.1:8000';
  } else {
    // When hosted on Static HF Space or served via cloud FastAPI
    defaultApiBase = 'https://inferroute-977n.onrender.com';
  }
}

export const API_BASE = (import.meta.env?.VITE_API_BASE as string) || (typeof window !== 'undefined' && localStorage.getItem('API_BASE')) || defaultApiBase;

