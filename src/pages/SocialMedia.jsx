import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import { Box, Button, CircularProgress, FormControl, InputLabel, MenuItem, Select, Typography } from '@mui/material';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import axios from 'axios';
import { Bar, Line } from 'react-chartjs-2';
import { Chart, CategoryScale, LinearScale, BarController, BarElement, LineController, LineElement, PointElement, Tooltip } from 'chart.js';
import { TypeAnimation } from 'react-type-animation';

const SocialMedia = () => {
  const [filteredPredictions, setFilteredPredictions] = useState([]);
  const [isDateFilterChecked, setIsDateFilterChecked] = useState(false);
  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(new Date());
  const [keyword, setKeyword] = useState('');
  const [isLoading, setIsLoading] = useState(false); // Loading state
  const [youtubePredictions, setYoutubePredictions] = useState([]);
  const [redditPredictions, setRedditPredictions] = useState([]);
  const [showOutput, setShowOutput] = useState(false); // State to control output visibility
  const [resultsToShow, setResultsToShow] = useState(5); // Number of results to show in the table
  const [selectedData, setSelectedData] = useState('all'); // Selected data filter

  const fetchPredictions = () => {
    // Fetch predictions from the Flask API
    axios
      .post('http://localhost:5000/social', {
        start_date: startDate.toISOString().slice(0, 10),
        end_date: endDate.toISOString().slice(0, 10),
        keyword,
      })
      .then((response) => {
        console.log(response.data.youtube_predictions); // Log the youtube_predictions to the console
        console.log(response.data.reddit_predictions); // Log the reddit_predictions to the console
        setYoutubePredictions(response.data.youtube_predictions);
        setRedditPredictions(response.data.reddit_predictions);
        setShowOutput(true); // Set showOutput to true when predictions are received
        setIsLoading(false); // Set loading state to false after receiving predictions
        
        // Filter predictions based on the selected data filter
        let filteredPredictions = [];
        if (selectedData === 'source1') {
          filteredPredictions = response.data.reddit_predictions;
        } else if (selectedData === 'source2') {
          filteredPredictions = response.data.youtube_predictions;
        } else {
          filteredPredictions = [...response.data.youtube_predictions, ...response.data.reddit_predictions];
        }
        setFilteredPredictions(filteredPredictions);
      })
      .catch((error) => {
        console.log(error);
        setIsLoading(false); // Set loading state to false if an error occurs
      });
  };
  

  const handleSubmit = () => {
    setIsLoading(true); // Set loading state to true
    setShowOutput(false); // Reset showOutput to false before making the API call
    Chart.register(CategoryScale, LinearScale, BarController, BarElement, LineController, LineElement, PointElement, Tooltip);
    fetchPredictions(); // Fetch predictions from the Flask API
  };

  const handleDataChange = (event) => {
    setSelectedData(event.target.value); // Update selectedData state with the selected value from the dropdown
  };

  // Extract sentiment labels and counts from youtubePredictions or redditPredictions
  const getSentimentData = () => {
    let predictions;
    if (selectedData === 'source1') {
      predictions = redditPredictions;
    } else if (selectedData === 'source2') {
      predictions = youtubePredictions;
    } else {
      predictions = [...youtubePredictions, ...redditPredictions];
    }

    if (!predictions) {
      return null; // Return null or an empty dataset when predictions are undefined
    }

    const sentiments = predictions.map((prediction) => prediction.sentiment);
    const sentimentCounts = {};

    for (let sentiment of sentiments) {
      if (sentiment in sentimentCounts) {
        sentimentCounts[sentiment] += 1;
      } else {
        sentimentCounts[sentiment] = 1;
      }
    }

    return {
      labels: Object.keys(sentimentCounts),
      datasets: [
        {
          label: 'Sentiment Count',
          data: Object.values(sentimentCounts),
          backgroundColor: [
            'rgba(255, 99, 132, 0.6)',
            'rgba(54, 162, 235, 0.6)',
            'rgba(75, 192, 192, 0.6)',
            'rgba(255, 206, 86, 0.6)',
            'rgba(153, 102, 255, 0.6)',
          ],
        },
      ],
    };
  };

  // Extract reason labels and counts from youtubePredictions or redditPredictions
// Extract reason labels and counts from youtubePredictions or redditPredictions
const getReasonData = () => {
  let predictions;
  if (selectedData === 'source1') {
    predictions = redditPredictions;
  } else if (selectedData === 'source2') {
    predictions = youtubePredictions;
  } else {
    predictions = [...youtubePredictions, ...redditPredictions];
  }

  if (!predictions) {
    return null; // Return null or an empty dataset when predictions are undefined
  }

  const reasons = predictions.flatMap((prediction) => prediction.reasons);
  const reasonCounts = {};

  for (let reason of reasons) {
    if (reason in reasonCounts) {
      reasonCounts[reason] += 1;
    } else {
      reasonCounts[reason] = 1;
    }
  }

  const labels = Object.keys(reasonCounts);
  const datasets = [
    {
      label: 'Reason Count',
      data: Object.values(reasonCounts),
      backgroundColor: [
        'rgba(255, 99, 132, 0.6)',
        'rgba(54, 162, 235, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(255, 206, 86, 0.6)',
        'rgba(153, 102, 255, 0.6)',
      ],
    },
  ];

  return {
    labels,
    datasets,
  };
};

  return (
    <>
      <Navbar />
      <Box sx={{ margin: '10% 10%' }}>
      <TypeAnimation
          sequence={[
            'Enter The Keyword',
            2000, // Waits 2s
            () => ' ', // Clears the text
            100, // Waits 2s
            'Enter The Keyword', // Types the text again
            () => {
              console.log('Sequence completed'); // Place optional callbacks anywhere in the array
            },
          ]}
          wrapper="span"
          cursor={true}
          repeat={Infinity}
          style={{ fontSize: '2em', display: 'inline-block' }}
        />
        <Box sx={{ backgroundColor: 'white', borderRadius: '10px', display: 'flex', padding: '10%' }}>
          <input
            id="sainput"
            placeholder="Example: Airtel"
            style={{backgroundColor:'#E0E4FD', fontSize: '100%', paddingLeft: '20px' }}
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
          />
          <Box sx={{ display: 'flex', flexDirection: 'column', marginLeft: '20px', marginRight: '20px' }}>
            <DatePicker selected={startDate} onChange={setStartDate} />
            <DatePicker selected={endDate} onChange={setEndDate} />
          </Box>
          <Box>
          <Button variant="contained" onClick={handleSubmit} disabled={isLoading}>
              Analyse
          </Button>

          </Box>
        </Box>

        {isLoading ? (
          <Box sx={{ textAlign: 'center', marginTop: '20px' }}>
            <CircularProgress color="primary" />
          </Box>
        ) : (
          showOutput && ( // Render output only if showOutput is true
            <Box sx={{ marginTop: '20px' }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <Typography variant="h6">Sentiment Analysis</Typography>
                <FormControl sx={{ minWidth: 120 }}>
                  <InputLabel>Data</InputLabel>
                  <Select value={selectedData} onChange={handleDataChange}>
                    <MenuItem value="all">All</MenuItem>
                    <MenuItem value="source1">Reddit</MenuItem>
                    <MenuItem value="source2">Youtube</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              <div style={{ display: 'flex', flexDirection: 'row' }}>
                <div style={{ flex: 1, marginRight: '20px' }}>
                  <Bar data={getSentimentData()} options={{ responsive: true }} />
                </div>
                <div style={{ flex: 1 }}>
                <Line
                    data={getReasonData()}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      elements: {
                        line: {
                          borderWidth: 8, // Increase line width
                          fill: true, // Disable shading

                        },
                        point: {
                          radius: 4, // Increase point size
                        },
                      },
                    }}
                  />
                </div>
              </div>
              <table style={{ marginTop: '20px', border: '1px solid #ccc' }}>
                <thead style={{ backgroundColor: '#9aa4e7'}}>
                  <tr>
                    <th>Text</th>
                    <th>Sentiment</th>
                    <th>Reason</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody style={{ backgroundColor: '#E0E4FD' }}>
                  {youtubePredictions.map((prediction, index) => (
                    <tr key={index}>
                      <td>{prediction.text}</td>
                      <td align="center">{prediction.sentiment}</td>
                      <td align="center">{prediction.reasons}</td>
                      <td align="center">{prediction.confidence}</td>
                    </tr>
                  ))}
                  {redditPredictions.map((prediction, index) => (
                    <tr key={index}>
                      <td>{prediction.text}</td>
                      <td align="center">{prediction.sentiment}</td>
                      <td align="center">{prediction.reasons}</td>
                      <td align="center">{prediction.confidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
          )
        )}
      </Box>
    </>
  );
};

export default SocialMedia;
