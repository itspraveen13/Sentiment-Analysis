import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import { Box, Button, CircularProgress, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import axios from 'axios';
import Paper from '@mui/material/Paper';
import { TypeAnimation } from 'react-type-animation';
import gifs from '../assets/img/sen.gif';

const SentimentAnalysis = () => {
  const [file, setFile] = useState(null);
  const [text, setText] = useState('');
  const [result, setResult] = useState([]);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAudio, setIsAudio] = useState(false);
  const [language, setLanguage] = useState('Tamil');

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setIsAudio(e.target.files[0].type.startsWith('audio'));
  };

  const handleTextAnalyseClick = () => {
    if (!text) {
      setError('Please enter a review');
      return;
    }
    setError(null);
    setIsLoading(true);

    axios
      .post('http://localhost:5000/text', { text })
      .then((response) => {
        console.log(response.data);
        setResult(
          response.data.predictions.map((predictions) => ({
            text: predictions.text,
            sentiment: predictions.sentiment,
            reason: predictions.reasons,
            confidence: predictions.confidence, // Add confidence to the result
          }))
        );
        setIsLoading(false);
      })
      .catch((error) => {
        console.error(error);
        setError('Error occurred while analyzing the text');
        setIsLoading(false);
      });
  };

  const handleFileAnalyseClick = () => {
    if (!file) {
      setError('Please upload a file');
      return;
    }
    setError(null);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', language);

    if (isAudio) {
      axios
        .post('http://localhost:5000/audio', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })
        .then((response) => {
          console.log(response.data);
          setResult(
            response.data.predictions.map((predictions) => ({
              text: predictions.text,
              sentiment: predictions.sentiment,
              reason: predictions.reasons,
              confidence: predictions.confidence, // Add confidence to the result
            }))
          );
          setIsLoading(false);
        })
        .catch((error) => {
          console.error(error);
          setError('Error occurred while analyzing the audio file');
          setIsLoading(false);
        });
    } else {
      axios
        .post('http://localhost:5000/dataset', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })
        .then((response) => {
          console.log(response.data);
          setResult(
            response.data.predictions.map((predictions) => ({
              text: predictions.text,
              sentiment: predictions.sentiment,
              reason: predictions.reasons,
              confidence: predictions.confidence, // Add confidence to the result
            }))
          );
          setIsLoading(false);
        })
        .catch((error) => {
          console.error(error);
          setError('Error occurred while analyzing the file');
          setIsLoading(false);
        });
    }
  };

  const handleCheckboxChange = (e) => {
    setShowDropdown(e.target.checked);
  };

  return (
    <>
      <Navbar />
      <Box sx={{ margin: '6%' }}>
        {/* <h3 style={{ textAlign: 'left' }}>Type your review or Upload Dataset</h3> */}

        <Box sx={{ backgroundColor: 'white', borderRadius: '10px', padding: '10%', width: '80%' }}>
          <TypeAnimation
            sequence={[
              'Type Your Text or Upload Dataset',
              2000, // Waits 2s
              () => ' ', // Clears the text
              100, // Waits 2s
              '', // Types the text again
              () => {
                console.log('Sequence completed'); // Place optional callbacks anywhere in the array
              },
            ]}
            wrapper="span"
            cursor={true}
            repeat={Infinity}
            style={{ fontSize: '2em', display: 'inline-block', color: '#121e36' }}
          />
          <input
            id="sainput"
            placeholder="Example: Jio Fiber Network is Fast"
            style={{ fontSize: '100%', paddingLeft: '20px', backgroundColor: '#E0E4FD' }}
            onChange={handleTextChange}
          />
          <Box sx={{ display: 'flex', justifyContent: 'center', margin: '10px' }}>
            <Button variant="contained" sx={{ height: '50px', backgroundColor: '#E0E4FD', color: 'black' }} onClick={handleTextAnalyseClick}>
              Analyse Text
            </Button>
            <h3 style={{ margin: '0 10px', color: 'black' }}>OR</h3>
            {file ? (
              <Button variant="contained" sx={{ height: '50px', backgroundColor: '#E0E4FD', color: 'black' }} onClick={handleFileAnalyseClick}>
                Start Analyse
              </Button>
            ) : (
              <label htmlFor="file">
                <Button component="span" variant="contained" sx={{ height: '50px', backgroundColor: '#E0E4FD', color: 'black' }}>
                  Upload File
                </Button>
                <input type="file" id="file" name="file" style={{ display: 'none' }} onChange={handleFileChange} />
                <Stack direction={'column'}>
                  <Box sx={{ padding: '10px 5px' }}>
                    <input
                      style={{ width: '50px', transform: 'scale(2)' }}
                      type="checkbox"
                      id="audioCheckbox"
                      checked={isAudio}
                      onChange={(e) => setIsAudio(e.target.checked)}
                    />
                    <label htmlFor="audioCheckbox" style={{ color: 'black' }}>
                      Audio file
                    </label>
                  </Box>
                  {isAudio && (
                    <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                      <option value="Tamil">Tamil</option>
                      <option value="French">French</option>
                      <option value="Spanish">Spanish</option>
                      <option value="German">German</option>
                      <option value="Italian">Italian</option>
                    </select>
                  )}
                </Stack>
              </label>
            )}
          </Box>
          <img
            src={gifs}
            alt="GIF"
            style={{
              width: '500px',
              height: '600px',
              position: 'absolute',
              top: '35%',
              right: '10px',
              transform: 'translateY(-50%)',
            }}
          />
          <br></br>
          <br></br>
          {error && <p style={{ color: 'red' }}>{error}</p>}
          {isLoading ? (
            <CircularProgress />
          ) : result.length ? (
            <TableContainer component={Paper}>
  <Table sx={{ minWidth: 650 }} aria-label="caption table">
    <TableHead>
      <TableRow style={{ backgroundColor: '#9aa4e7' }}>
        <TableCell style={{ fontWeight: 'bold' }}>Review</TableCell>
        <TableCell align="center" style={{ fontWeight: 'bold' }}>
          Sentiment
        </TableCell>
        <TableCell align="center" style={{ fontWeight: 'bold' }}>
          Reason
        </TableCell>
        <TableCell align="center" style={{ fontWeight: 'bold' }}>
          Confidence
        </TableCell>
        {/* Add Confidence column */}
      </TableRow>
    </TableHead>
    <TableBody>
      {result.map((res, index) => (
        <TableRow key={index} style={{ backgroundColor: '#E0E4FD' }}>
          <TableCell>{res.text}</TableCell>
          <TableCell align="center">{res.sentiment}</TableCell>
          <TableCell align="center">{res.reason}</TableCell>
          <TableCell align="center">{res.confidence}</TableCell>
          {/* <TableCell>I would like to thank the customer service as they provided good support</TableCell>
          <TableCell align="center">Positive</TableCell>
          <TableCell align="center">good support</TableCell>
          <TableCell align="center">89.02</TableCell> */}
          {/* Display Confidence */}
        </TableRow>
      ))}
    </TableBody>
    <style>
      {`
      table {
        border-collapse: separate;
        border-spacing: 0 8px;
      }
      
      tbody tr:first-child {
        border-top: none;
      }
      
      tbody tr:last-child {
        border-bottom: none;
      }
      
      td {
        border-bottom: 1px solid #ccc;
        border-right: 1px solid #ccc;
        border-left: 1px solid #ccc;

      }
      `}
    </style>
  </Table>
</TableContainer>

          ) : null}
        </Box>
      </Box>
    </>
  );
};

export default SentimentAnalysis;
