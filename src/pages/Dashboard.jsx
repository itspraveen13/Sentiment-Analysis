import React, { useState } from 'react';
import axios from 'axios';
import '../assets/css/cp.css';
import Navbar from '../components/Navbar';
import { Box, CircularProgress } from '@mui/material';
import '../assets/css/meme.css'

const MemePrediction = () => {
  const [fileNames, setFileNames] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState('');
  const [responseMessage, setResponseMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const files = event.target.files;
    const names = [];
    for (let i = 0; i < files.length; i++) {
      names.push(files[i].name);
    }
    setFileNames(names);
    setSelectedFile(URL.createObjectURL(files[0]));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    const fileInput = document.querySelector('#file');
    formData.append('file', fileInput.files[0]);

    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/meme', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data.sentiment);
      setResponseMessage('');
      setIsLoading(false);
      setSelectedFile(URL.createObjectURL(fileInput.files[0])); // Display the image after receiving the output
    } catch (error) {
      console.error(error);
      setError('Error occurred while analyzing');
      setIsLoading(false);
    }
  };

  return (
    <>
      <Navbar />
      <div className="formbold-main-wrapper">
        <div className="formbold-form-wrapper">
          <form onSubmit={handleSubmit}>
            <div className="mb-6 pt-4">
              {isLoading && (
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
                  <CircularProgress />
                </div>
              )}
              <div className="formbold-mb-5 formbold-file-input">
                <input type="file" name="file" id="file" onChange={handleFileChange} multiple />
                <label htmlFor="file">
                  <div>
                    <span className="formbold-drop-file"> Drop Image here </span>
                    <span className="formbold-or"> Or </span>
                    <span className="formbold-browse"> Browse </span>
                  </div>
                </label>
              </div>

              {fileNames.length > 0 && (
                <div className="formbold-file-list formbold-mb-5">
                  {fileNames.map((name, index) => (
                    <div className="formbold-file-item" key={index}>
                      <span className="formbold-file-name">{name}</span>
                    </div>
                  ))}
                </div>
              )}

              <div>
                <button className="formbold-btn w-full" style={{ fontWeight: '400' }} type="submit">
                  Send File
                </button>
              </div>
            </div>
          </form>
          {result && selectedFile && (
          <div className="result">
            <div className="image-sentiment-container">
              <div className="image-container">
                <img src={selectedFile} alt="Uploaded" className="preview-image" />
              </div>
              <div className="sentiment">
                <span className="prediction-label">Sentiment:
                {result}</span>
              </div>
            </div>
          </div>
        )}
        {error && <div className="error">{error}</div>}

        </div>
      </div>
    </>
  );
};

export default MemePrediction;
