import { Box, Button } from '@mui/material';
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import Lottie from 'lottie-react';
import animationData from '../assets/js/Animation.json';
import BGimg from '../assets/img/BG_IMG.png'
import Footer from '../component/Footer.jsx'

const Meme = () => {
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [results, setResults] = useState([]);
    const [responseMessage, setResponseMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [sentimentGroups, setSentimentGroups] = useState({ positive: [], negative: [] });
    const [uploadedFile, setUploadedFile] = useState(null);
    const [fileNames, setFileNames] = useState([]);
    const resultBoxRef = useRef(null);
    const resultBoxRef1 = useRef(null);
    const fileInputRef = useRef(null);

    const handleScrollIntoView = () => {
      if (resultBoxRef.current) {
        resultBoxRef.current.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
        });
      }
      if (resultBoxRef1.current) {
        resultBoxRef1.current.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }
    };
  
    useEffect(() => {
      // Scroll into view when the component mounts and conditions are met
      if (results) {
        handleScrollIntoView();
      }
    }, [results]);

    useEffect(() => {
      if (results.length === 0) {
        window.scrollTo(0, 0);
      } else {
        document.body.style.overflow = 'auto';
      }
    
      return () => {
        document.body.style.overflow = 'auto';
      };
    }, [results]);

    const handleSubmit = async (files) => {
      const formDataArray = [];
    
      setIsLoading(true);
      setError(null);
    
      const names = files.map((file) => file.name);
      setFileNames(names);

      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        formDataArray.push(formData);
      }
    
      try {
        const responses = await Promise.all(
          formDataArray.map((formData) =>
            axios.post('http://localhost:5000/meme', formData, {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            })
          )
        );
    
        const sentimentResults = responses.map((response) => response.data.sentiment);
        setResults(sentimentResults);
    
        const positiveImages = [];
        const negativeImages = [];
    
        for (let i = 0; i < files.length; i++) {
          if (sentimentResults[i] === 'Positive') {
            positiveImages.push(files[i]);
          } else {
            negativeImages.push(files[i]);
          }
        }
    
        setSentimentGroups({ positive: positiveImages, negative: negativeImages });
    
        console.log(responses);
        setResponseMessage('');
        setIsLoading(false);
        setSelectedFiles(files);
      } catch (error) {
        console.log('error:', error);
        setError('Error occurred while analyzing');
        setIsLoading(false);
      }
    };
    
    const handleClearResults = () => {
      window.scrollTo(0, 0);
      setResults([]);
      setUploadedFile(null);
  }

    const handleFileSelect = (event) => {
      const files = Array.from(event.target.files || []);
      if (files.length > 0) {
        handleSubmit(files);
        setUploadedFile(files);
      }
      event.target.value = '';
    };
    return (
        <>
              {uploadedFile ? (
                <>
                  <Box id="custombox3">
                    <h2 id='boxheader'>Uploaded File</h2>
                    {fileNames.map((fileName, index) => (
                      <p key={index}>File{index+1} Name: {fileName}</p>
                    ))}
                    <Box style={{ textAlign: 'center', marginTop: '5%' }}>
                      <Button
                        variant="outlined"
                        color="secondary"
                        onClick={handleClearResults}
                      >
                        Remove File
                      </Button>
                    </Box>
                  </Box>
                  <div id='tex' style={{ position: "absolute", top: "350px" }}>
                    <img style={{ width: "1700px" }} src={BGimg} alt="img" />
                  </div>
                </>
              ) : (
                <>
                <Box id="custombox2">
                  <h2 id='boxheader'>Meme Analysis</h2>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".png,.jpg,.jpeg"
                      multiple
                      style={{ display: 'none' }}
                      onChange={handleFileSelect}
                    />
                    <div id="dropbox" style={{ textAlign: 'center', padding: '20px' }}>
                      <Button
                        variant="outlined"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        Upload Images
                      </Button>
                    </div>
                  <h5 style={{ textAlign: "center", paddingTop: "30px", fontWeight: "400" }}>
                    (For File Upload only png, jpeg, jpg format files are accepted and the image should contain atleast one Text)
                  </h5>
                  </Box>
                  <div id='tex' style={{ position: "absolute", top: "550px" }}>
                    <img style={{ width: "1700px" }} src={BGimg} alt="img" />
                  </div>
                </>
              )}
              
            {isLoading && 
            (
              <div style={{ height: '150px', marginTop: '5%' }}>
                  <Lottie animationData={animationData} loop={true} />
                </div>
              )}
             {results.length > 0 && (
                <Box ref={resultBoxRef} style={{ display: 'flex', marginTop: '4%', justifyContent: 'center' }}>
                  {results.length === 1 ? (
                    <div style={{ textAlign: 'center' }}>
                      <img
                        style={{ height: '400px', width: '500px' }}
                        src={URL.createObjectURL(selectedFiles[0])}
                        alt="Uploaded Meme"
                      />
                      <h2 style={{ marginTop: '10px' }}>Sentiment: {results[0]}</h2>
                    </div>
                  ) : (
                    <Box ref={resultBoxRef1} >
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '28%' }}>
                        <div style={{ flex: 1, padding:'3%', border: '1px solid #ddd', borderRadius: '5px', marginRight: '2%' }}>
                          <div style={{ textAlign: 'center', marginBottom: '5%' }}>
                            <h2>Positive Sentiments</h2>
                          </div>
                          {sentimentGroups.positive.map((fileObj, index) => (
                            <div key={index} className="image-sentiment-container" style={{ }}>
                              <div className="image-container"  style={{ marginBottom: '2%' }}>
                                <img
                                  style={{ height: '400px', width: '500px' }}
                                  src={URL.createObjectURL(fileObj)}
                                  alt={`Uploaded Meme ${index + 1}`}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                        <div style={{ flex: 2, padding: '3%', border: '1px solid #ddd', borderRadius: '5px' }}>
                          <div style={{ textAlign: 'center', marginBottom: '5%' }}>
                            <h2>Negative Sentiments</h2>
                          </div>
                          {sentimentGroups.negative.map((fileObj, index) => (
                            <div key={index} className="image-sentiment-container" style={{  }}>
                              <div style={{ marginBottom: '2%' }}>
                                <img
                                  style={{ height: '400px', width: '500px' }}
                                  src={URL.createObjectURL(fileObj)}
                                  alt={`Uploaded Meme ${index + 1}`}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </Box>    
                  )}
                </Box>
              )}
             <h3 style={{ color: 'red', marginLeft: "40%", marginTop: '3%'}}>{error}</h3>
             <Box style={{ marginTop: results.length >= 1 ? '13%' : '30%' }}>
                  <Footer />
             </Box>
        </>
    )
}

export default Meme
