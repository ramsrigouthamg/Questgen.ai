import axios from 'axios';

const apiUrl = 'http://localhost:8000/predict'; 

export const fetchData = async () => {
  try {
    const response = await axios.get(`${apiUrl}/data`);
    return response.data;
  } catch (error) {
    throw error;
  }
};