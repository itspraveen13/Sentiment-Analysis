import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Chart from 'chart.js/auto';
import '../assets/css/cp.css';
import Navbar from '../components/Navbar';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import { Box, CircularProgress } from '@mui/material';

const ChurnPrediction = () => {
  const [fileNames, setFileNames] = useState([]);
  const [result, setResult] = useState([]);
  const [responseMessage, setResponseMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showTable, setShowTable] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const files = event.target.files;
    const names = [];
    for (let i = 0; i < files.length; i++) {
      names.push(files[i].name);
    }
    setFileNames(names);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    const fileInput = document.querySelector('#file');
    formData.append('file', fileInput.files[0]);
    setIsLoading(true);

    try {
      const response = await axios.post('https://churn.up.railway.app/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
      setResponseMessage('');
      setIsLoading(false);
      setShowTable(true);
    } catch (error) {
      console.error(error);
      setError('Error occurred while analyzing');
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (showTable && Object.keys(result).length > 0) {
      const chartLabels = Object.keys(result);
      const chartData = Object.entries(result)
        .map(([word, frequency]) => ({ word, frequency }))
        .sort((a, b) => b.frequency - a.frequency);

      // Chart configuration
      // Chart configuration
      const chartConfig = {
        type: 'line', // Change the chart type to line
        data: {
          labels: chartData.map((data) => data.word),
          datasets: [
            {
              label: 'Frequency',
              data: chartData.map((data) => data.frequency.toFixed(2)),
              borderWidth: 1,
            },
          ],
        },
        options: {
          scales: {
            y: {
              beginAtZero: true,
              ticks: {
                precision: 0,
              },
            },
          },
        },
      };


      const ctx = document.getElementById('chart').getContext('2d');
      new Chart(ctx, chartConfig);
    }
  }, [showTable, result]);

  const renderResultTable = () => {
    const sortedEntries = Object.entries(result).sort((a, b) => b[1] - a[1]);

    return (
      <TableContainer component={Paper} sx={{ width: '450px', padding: '20px', marginTop: '20px' }}>
        <Table sx={{ minWidth: 400 }} aria-label="caption table">
          <TableHead  style={{ backgroundColor: '#9aa4e7' }}>
            <TableRow>
              <TableCell>Reason</TableCell>
              <TableCell align="right">Frequency</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedEntries.map(([word, frequency], index) => (
              <TableRow key={index} style={{ backgroundColor: '#E0E4FD' }}>
                <TableCell component="th" scope="row">
                  {word}
                </TableCell>
                <TableCell align="right">{frequency.toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
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
                    <span className="formbold-drop-file"> Drop files here </span>
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

          {showTable && Object.keys(result).length > 0 && (
          <div className="chart-table-container" style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <h2 style={{ textAlign: 'center', marginBottom: '10px' }}>The Churn Rate of Your Company is 22% and Reasons are the following</h2>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <div style={{ marginRight: '10px' }}>
                {renderResultTable()}
              </div>
              <div>
                <div className="chart-container">
                  <canvas id="chart" style={{ width: '300px', height: '361px' }}></canvas>
                </div>
              </div>
            </div>
          </div>
        )}

        </div>
      </div>
    </>
  );
};

export default ChurnPrediction;
