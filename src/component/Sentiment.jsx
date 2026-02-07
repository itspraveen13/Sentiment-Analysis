import React, { useState, useEffect, useRef } from 'react';
import '../assets/css/common.css'
import KeyboardVoiceIcon from '@mui/icons-material/KeyboardVoice';
import * as XLSX from 'xlsx';
import { ref,getStorage, listAll, uploadBytes, getDownloadURL } from "firebase/storage";
import { storage } from '../assets/js/config.js';
import BGimg from '../assets/img/BG_IMG.png'
import Footer from '../component/Footer.jsx'
import {
  Box,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  MenuItem,
  IconButton,
  Menu,
} from '@mui/material';
import axios from 'axios';
import Paper from '@mui/material/Paper';
import { scaleLinear } from 'd3-scale';
import Grid from '@mui/material/Grid'; 
import { Chart } from 'react-google-charts';
import WordCloud from 'wordcloud';
import SentimentSatisfiedAltIcon from '@mui/icons-material/SentimentSatisfiedAlt';
import SentimentVeryDissatisfiedIcon from '@mui/icons-material/SentimentVeryDissatisfied';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import html2canvas from 'html2canvas';
import { saveAs } from 'file-saver';
import jsPDF from 'jspdf';
import ClearIcon from '@mui/icons-material/Clear';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown'; // Import the arrow icon
import Lottie from 'lottie-react';
import animationData from '../assets/js/Animation.json';
import errorAnimationData from '../assets/js/Error Animation.json';

const Sentiment = () => {
    const [file, setFile] = useState(null);
    const [text, setText] = useState('');
    const [result, setResult] = useState([]);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [excelData, setExcelData] = useState(null);
    const [countryData, setCountryData] = useState([]);
    const [positiveCounts, setPositiveCounts] = useState({});
    const [negativeCounts, setNegativeCounts] = useState({});
    const [analysisCompleted, setAnalysisCompleted] = useState(false);
    const [genderChartData, setgenderChartData] = useState([]);
    const [totalPositiveCounts, setTotalPositiveCounts] = useState(0);
    const [totalNegativeCounts, setTotalNegativeCounts] = useState(0);
    const [ageChartData, setAgeChartData] = useState([]);
    const [titleChartData, setTitleChartData] = useState([]);
    const [sentimentChartData, setSentimentChartData] = useState([]);
    const [countryPlotData, setCountryPlotData] = useState([]);
    const [wordCloudData, setWordCloudData] = useState({});
    const [geoChartData, setGeoChartData] = useState([]);
    const [insight, setInsights] = useState(null);
    const [textOutput, setTextOutput] = useState([]);
    const [negChart, setNegChart] = useState([]);
    const [posChart, setPosChart] = useState([]);
    const [overAllPosChartData, setOverAllPosChartData] = useState([]);
    const [overAllNegChartData, setOverAllNegChartData] = useState([]);
    const [pieStartAngle, setPieStartAngle] = useState('null');
    const [posSenWidth, setPosSenWidth] = useState('null');
    const [negSenWidth, setNegSenWidth] = useState('null');
    const [pieEndAngle, setPieEndAngle] = useState('null');
    const [sentimentFilter, setSentimentFilter] = useState('Both'); // 'Positive', 'Negative', or 'Both'
    const [confidenceFilter, setConfidenceFilter] = useState('All'); // '40-60', '61-80', '81-100', or 'All'
    const [sentimentMenuAnchor, setSentimentMenuAnchor] = useState(null);
    const [confidenceMenuAnchor, setConfidenceMenuAnchor] = useState(null);
    const resultBoxRef = useRef(null);
    const resultBoxRef1 = useRef(null);
    const wordCloudCanvasRef = useRef(null);

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
      if (result.length >= 1 && analysisCompleted) {
        handleScrollIntoView();
      }
    }, [result, analysisCompleted]);

    const filteredResults = result.filter((res) => {
      if (sentimentFilter === 'Both' || res.sentiment === sentimentFilter) {
        if (confidenceFilter === 'All' || (
          res.confidence >= Number(confidenceFilter.split('-')[0]) &&
          res.confidence <= Number(confidenceFilter.split('-')[1])
        )) {
          return true;
        }
      }
      return false;
    });

    // Function to render the table rows
    const renderRows = () => {
      return filteredResults.map((res, index) => (
        <TableRow key={index}>
          <TableCell style={{ fontSize: '18px', width: '10%' }}>
            {highlightWords(res.text, res.reason, res.sentiment, res.words_list)}
          </TableCell>
          <TableCell align="center" style={{ fontSize: '16px', width: '1%' }}>
            {res.sentiment}
          </TableCell>
          <TableCell align="center" style={{ fontSize: '15px', width: '5%' }}>
            {res.title.length > 1 ? res.title.join(', ') : res.title[0]}
            {res.title.length === 0 ? "None" : ""}
          </TableCell>
          <TableCell align="center" style={{ fontSize: '16px', width: '1%' }}>
            {res.confidence}
          </TableCell>
        </TableRow>
      ));
    };

    const handleSaveAsPdf = () => {
      const mainGrid = document.getElementById('insightPDF');
      html2canvas(mainGrid)
        .then((canvas) => {
          const imgWidth = canvas.width;
          const imgHeight = canvas.height;
          const scaleFactor = 800 / imgWidth;
          const scaledWidth = imgWidth * scaleFactor;
          const scaledHeight = imgHeight * scaleFactor;
          const pdf = new jsPDF('p', 'pt', [scaledWidth,scaledHeight]);
          pdf.addImage(canvas, 'PNG', 0, 0, scaledWidth, scaledHeight);
          const pdfBlob = pdf.output('blob');
          saveAs(pdfBlob, 'Sentiment.pdf');
        });
    };
    
  const handleTextChange = (e) => {
    setText(e.target.value);
  };
  
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    handleFileAnalyseClick(e.target.files[0]);
  };

  function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

  const handleTextAnalyseClick = () => {
    if (!text) {
      setError('Please enter a review');
      return;
    }
    setError(null);
    setIsLoading(true);

    axios
    .post('http://localhost:5000/text_sentiment', { text })
    .then((response) => {
      console.log(response.data);
      const predictions = response.data.predictions;
  
      if (predictions.length === 1) {
        setResult(
          response.data.predictions.map((predictions) => ({
            text: predictions.text,
            sentiment: predictions.sentiment,
            reason: predictions.reasons,
            confidence: predictions.confidence,
            titles: predictions.titles,
            title_words: predictions.title_words,
          }))
          );
      }
      setTextOutput(response.data.prediction);
      setIsLoading(false);
      setAnalysisCompleted(true);
    })
    .catch((error) => {
      console.error(error);
      setError('Error occurred while analyzing the text');
      setIsLoading(false);
      setResult([]);
    });  
  };

  const handleFileAnalyseClick = (file) => {
    if (!file) {
      setError('Please upload a file');
      return;
    }
    setError(null);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    const isAudio = file.type.startsWith('audio');
    if (isAudio) {
      // Create a FormData object and append the selected file to it.
      const formData = new FormData();
      formData.append('file', file);
      // Make the axios POST request for audio analysis.
      axios
        .post('http://localhost:5000/audio_sentiment', formData, {
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
              confidence: predictions.confidence,
              title_words: predictions.title_words,
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
        .post('http://localhost:5000/file_sentiment', formData, {
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
              title: predictions.titles,
              country: predictions.country,
              age: predictions.age,
              gender: predictions.gender,
              confidence: predictions.confidence,
              coordinates: predictions.coordinates, 
              words_list: predictions.words_corresponding_to_title,
            }))
          );  
          setInsights(response.data.insight);
          setPositiveCounts(response.data.positive_counts);
          setNegativeCounts(response.data.negative_counts);
          const totalPositive = Object.values(positiveCounts).reduce(
            (acc, count) => acc + count,
            0
          );
          const totalNegative = Object.values(negativeCounts).reduce(
            (acc, count) => acc + count,
            0
          );
          setTotalPositiveCounts(response.data.total_positive);
          setTotalNegativeCounts(response.data.total_negative);
          console.log("pos: ", totalPositive)
          const titleCounts = {};
          response.data.predictions.forEach((prediction) => {
              prediction.titles.forEach((title) => {
                  if (titleCounts[title]) {
                      titleCounts[title]++;
                  } else {
                      titleCounts[title] = 1;
                  }
              });
          });
          
          const titleChartData = [['Title', 'Count', { role: 'style' }]];
          Object.keys(titleCounts).forEach((title) => {
              titleChartData.push([title, titleCounts[title], getRandomColor()]);
          });

          const reasons = response.data.predictions.flatMap((prediction) => prediction.titles).filter(Boolean);
          const reasonCounts = {};          
          // Count the frequency of each reason
          for (const reason of reasons) {
            if (reason in reasonCounts) {
              reasonCounts[reason]++;
            } else {
              reasonCounts[reason] = 1;
            }
          }
          
          const wordCloudData = Object.entries(reasonCounts).map(([reason, count]) => ({
                        text: reason,
                        value: count,
                      }));

            const posSenCount = response.data.insight.positive_sentiment_count;
            const negSenCount = response.data.insight.negative_sentiment_count;
            const totalSen = posSenCount + negSenCount;
            const positivePercentage = (posSenCount / totalSen) * 100;
            const negativePercentage = (negSenCount / totalSen) * 100;
            const posEmpty = 100 - positivePercentage;
            const negEmpty = 100 - negativePercentage;
            const positiveCircle = [
              ['Sentiment', 'Percentage'],
              ['Positive', positivePercentage],
              ['Negative', posEmpty],
            ];
            const negativeCircle = [
              ['Sentiment', 'Percentage'],
              ['Negative', negativePercentage],
              ['Positive', negEmpty],
            ];
            
            const posChartData = [
              ['Sentiment', 'Count'],
              ['Positive', positivePercentage],
              ['Negative', posEmpty],
            ];

            const negChartData = [
              ['Sentiment', 'Count'],
              ['Negative', negativePercentage],
              ['Positive', negEmpty],
            ];

            const startAngle = 0; 
            const endAngleFirstChart = startAngle + (positivePercentage / 100) * 360; 
            const engAngleSecondChart = startAngle + (negativePercentage / 100) * 360;
                
            setPieEndAngle(engAngleSecondChart);
            setPieStartAngle(endAngleFirstChart);
            setOverAllPosChartData(posChartData);
            setOverAllNegChartData(negChartData);
            setPosChart(positiveCircle);
            setNegChart(negativeCircle);
            setTitleChartData(titleChartData);
            setWordCloudData(wordCloudData);
            setPosSenWidth(positivePercentage.toFixed(2) + '%');
            setNegSenWidth(negativePercentage.toFixed(2) + '%');

            
          try{             
            const coordinatesData = response.data.predictions.map((predictions) => ({
              country: predictions.country,
              latitude: parseFloat(predictions.coordinates[0]), 
              longitude: parseFloat(predictions.coordinates[1]), 
            }));    
            setCountryData(coordinatesData);
            const countryCounts = {};
            response.data.predictions.forEach((item) => {
                const country = item.country;
                if (countryCounts[country]) {
                    countryCounts[country]++;
                } else {
                    countryCounts[country] = 1;
                }
            });
            const countryPlotData = [['Country', 'Count', { role: 'style' }]];
            Object.keys(countryCounts).forEach((country) => {
                const color = getRandomColor();
                countryPlotData.push([country, countryCounts[country], `color: ${color}`]);
            });
            setCountryPlotData(countryPlotData);
          }catch(error){
            console.error("No data for country: ", error);
          }


          try{
            const genderCounts = {};
            response.data.predictions.forEach((prediction) => {
              const gender = prediction.gender;
              if (gender != 'none') {
                if (genderCounts[gender]) {
                  genderCounts[gender]++;
                } else {
                  genderCounts[gender] = 1;
                }
              }
            })         
            const genderChartData = Object.keys(genderCounts).map((gender) => ({
              label: gender,
              value: genderCounts[gender],
              style: {
                fill: gender === 'Male' ? '#41C0FB' : 'pink', // Customize colors based on gender
              },
            }));
            setgenderChartData(genderChartData);
          }catch(error){
            console.log("No data for gender: ",error);
          }

            try{
              const sentimentCounts = {
              male: { positive: 0, negative: 0 },
              female: { positive: 0, negative: 0 },
            };
            response.data.predictions.forEach((predictions) => {
              const gender = predictions.gender.toLowerCase();
              const sentiment = predictions.sentiment.toLowerCase();
              sentimentCounts[gender][sentiment]++;
            });
            // Prepare data for the chart
            const sentimentChartData = [
              ['Gender', 'Positive', 'Negative'],
              ['Male', sentimentCounts.male.positive, sentimentCounts.male.negative],
              ['Female', sentimentCounts.female.positive, sentimentCounts.female.negative],
            ];
            setSentimentChartData(sentimentChartData);
          }catch(error){
            console.log("No data for gender chart: ",error);
          }


            try{
            const positiveCountsObj = response.data.positive_counts;
            const negativeCountsObj = response.data.negative_counts;
            const geoChartData = [['Country', 'Positive', 'Negative']];
            // Iterate over the positive counts to populate the geochart data
            Object.keys(positiveCountsObj).forEach((country) => {
              const positiveCount = positiveCountsObj[country] || 0;
              const negativeCount = negativeCountsObj[country] || 0;
              geoChartData.push([country, positiveCount, negativeCount]);
            });
            setGeoChartData(geoChartData);
          }catch(error){
            console.log("Error no geo chart data: ",error);
          }


            try{
              const ageSentimentData = [
              ['Age Group', 'Positive', 'Negative'],
              ['15-25', 0, 0],
              ['25-35', 0, 0],
              ['35-45', 0, 0],
              ['45-55', 0, 0],
              ['55-65', 0, 0],
              ['65+', 0, 0],
            ];
            response.data.predictions.forEach((prediction) => {
              const age = prediction.age;
              const sentiment = prediction.sentiment;
              if (age!=0 && sentiment && (sentiment === 'Positive' || sentiment === 'Negative')) {
                if (age >= 15 && age <= 25) {
                  ageSentimentData[1][sentiment === 'Positive' ? 1 : 2]++;
                } else if (age > 25 && age <= 35) {
                  ageSentimentData[2][sentiment === 'Positive' ? 1 : 2]++;
                } else if (age > 35 && age <= 45) {
                  ageSentimentData[3][sentiment === 'Positive' ? 1 : 2]++;
                } else if (age > 45 && age <= 55) {
                  ageSentimentData[4][sentiment === 'Positive' ? 1 : 2]++;
                } else if (age > 55 && age <= 65) {
                  ageSentimentData[5][sentiment === 'Positive' ? 1 : 2]++;
                } else {
                  ageSentimentData[6][sentiment === 'Positive' ? 1 : 2]++;
                }
              }
            });
            setAgeChartData(ageSentimentData);
          }catch(error){
            console.log("No age data: ",error);
          }


          setIsLoading(false);
          setExcelData(response.data);
        })
        .catch((error) => {
          console.error(error);
          setError('Error occurred while analyzing the file');
          setIsLoading(false);
          setResult([]);
        });
    }
    setAnalysisCompleted(true);
    setFile(null);
  };

  function highlightWordsText(text, reasons, sentiment, title_words) {
    // Split the text into words
    const words = text
      .split(/\s+/) // Split by spaces
      .map((word) => word.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, '')); // Remove punctuation marks
  
    // Create a function to check if a word is in the reasons list
    const isReasonWord = (word) => reasons.includes(word.toLowerCase());
    
    // Create a function to check if a word is in the title_words list
    const isTitleWord = (word) => title_words.includes(word.toLowerCase());

    return words.map((word, index) => {
      // Check if the word is not empty and is a reason word or title word, then apply styling accordingly
      const wordStyle =
        word.trim() !== '' &&
        isReasonWord(word) &&
        sentiment === 'Positive'
          ? { backgroundColor: '#B9EACB', color: '#3A805B', border: '2px solid #4F7F65', padding: '2px', borderRadius: '3px' }
          : word.trim() !== '' &&
            isReasonWord(word) &&
            sentiment === 'Negative'
          ? { backgroundColor: '#FCD7DE', border: '2px solid #C60943', padding: '4px', color: '#C60943', borderRadius: '3px' }
          : word.trim() !== '' &&
            isTitleWord(word)
          ? { color: '#04BADE' }
          : {}; // No styling for other words or blank spaces

      // Add a space after each word except the last one
      const space = index < words.length - 1 ? ' ' : '';
  
      // Wrap the word in a span with the calculated style
      return (
        <span key={index} style={wordStyle}>
          {word}
          {space}
        </span>
      );
    });
  }

  function highlightWords(text, reasons, sentiment, title_words) {
    const words = text
      .split(/\s+/) // Split by spaces
      .map((word) => word.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, '')); // Remove punctuation marks
  
    // Create a function to check if a word is in the reasons list
    const isReasonWord = (word) => reasons.includes(word.toLowerCase());
    
    // Create a function to check if a word is in the title_words list
    const isTitleWord = (word) => title_words.includes(word.toLowerCase());

    return words.map((word, index) => {
      // Check if the word is not empty and is a reason word or title word, then apply styling accordingly
      const wordStyle =
        word.trim() !== '' &&
        isReasonWord(word) &&
        sentiment === 'Positive'
          ? { backgroundColor: '#B9EACB', color: '#3A805B', border: '2px solid #4F7F65', padding: '2px', borderRadius: '3px' }
          : word.trim() !== '' &&
            isReasonWord(word) &&
            sentiment === 'Negative'
          ? { backgroundColor: '#FCD7DE', border: '2px solid #C60943', padding: '4px', color: '#C60943', borderRadius: '3px' }
          : word.trim() !== '' &&
            isTitleWord(word)
          ? { padding:'10px', backgroundColor: '#83DCEE', border: '2px solid #04BADE', color: 'black', borderRadius: '30px' }
          : {}; // No styling for other words or blank spaces

      // Add a space after each word except the last one
      const space = index < words.length - 1 ? ' ' : '';
  
      // Wrap the word in a span with the calculated style
      return (
        <span key={index} style={wordStyle}>
          {word}
          {space}
        </span>
      );
    });
  }
  

  useEffect(() => {
    if (excelData) {
      handleExcelSave();
    }
  }, [excelData]);

  useEffect(() => {
    if (!wordCloudCanvasRef.current) {
      return;
    }
    const list = Array.isArray(wordCloudData)
      ? wordCloudData.map((item) => [item.text, item.value])
      : [];
    if (list.length === 0) {
      const ctx = wordCloudCanvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, wordCloudCanvasRef.current.width, wordCloudCanvasRef.current.height);
      }
      return;
    }
    WordCloud(wordCloudCanvasRef.current, {
      list,
      backgroundColor: 'rgba(0,0,0,0)',
      color: () => '#2b2b2b',
      rotateRatio: 0,
      rotationSteps: 1,
      gridSize: 10,
      weightFactor: (size) => Math.max(12, size * 6),
    });
  }, [wordCloudData]);

  const handleExcelDownloadClick = () => {
    if (excelData) {
      const wb = XLSX.utils.book_new(); // Create a new Excel workbook
  
      // Create a sheet for the combined counts data
      const countsSheetData = [];
  
      // Iterate through positiveCounts and add data to the counts sheet
      for (const country in positiveCounts) {
        const countryData = {
          Country: country,
          'Positive Count': positiveCounts[country],
          'Negative Count': 0,
        };
  
        if (country in negativeCounts) {
          countryData['Negative Count'] = negativeCounts[country];
          delete negativeCounts[country]; // Remove the country from negativeCounts
        }
  
        countsSheetData.push(countryData);
      }
  
      // Add remaining countries from negativeCounts
      for (const country in negativeCounts) {
        countsSheetData.push({
          Country: country,
          'Positive Count': 0,
          'Negative Count': negativeCounts[country],
        });
      }
    
      // Create a sheet for the sentiment data
      const sentimentSheetData = excelData.predictions.map((prediction) => ({
        Gender: prediction.gender,
        Age: prediction.age,
        Review: prediction.text,
        Country: prediction.country,
        Sentiment: prediction.sentiment,
        Reason: prediction.reasons ? prediction.reasons.join(', ') : '',
        Title: prediction.titles ? prediction.titles.join(', ') : '',
        Confidence: prediction.confidence,
      }));
  
      const countsSheet = XLSX.utils.json_to_sheet(countsSheetData);
      const sentimentSheet = XLSX.utils.json_to_sheet(sentimentSheetData);
  
      const currentDateIST = new Date().toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
      });
      
      // Splitting the date and time parts
      const parts = currentDateIST.split(', ');
      const datePart = parts[0].replace(/ /g, '-');
      const timePart = parts[1].replace(/:/g, '.');
            
      // Removing spaces between time and AM/PM and adding a hyphen
      const timeAMPM = timePart.replace(/\s+/g, '') + '-';
            
      const uniqueFileName = `${datePart}-${timeAMPM}sentiment_analysis_file.xlsx`;
            
      // Replacing slashes with periods
      const uniqueFileNameProcessed = uniqueFileName.replace(/\//g, '.');
           
      // Add both sheets to the same workbook
      XLSX.utils.book_append_sheet(wb, sentimentSheet, 'SentimentData');
      XLSX.utils.book_append_sheet(wb, countsSheet, 'CountryCounts');

      XLSX.writeFile(wb, uniqueFileNameProcessed);
    }
  };
  
  const handleExcelSave = async () => {
    if(excelData){
      const loggedUser = localStorage.getItem('username');
      if (loggedUser) {
      const wb = XLSX.utils.book_new(); // Create a new Excel workbook
  
      // Create a sheet for the combined counts data
      const countsSheetData = [];
  
      // Iterate through positiveCounts and add data to the counts sheet
      for (const country in positiveCounts) {
        const countryData = {
          Country: country,
          'Positive Count': positiveCounts[country],
          'Negative Count': 0,
        };
  
        if (country in negativeCounts) {
          countryData['Negative Count'] = negativeCounts[country];
          delete negativeCounts[country]; // Remove the country from negativeCounts
        }
  
        countsSheetData.push(countryData);
      }
  
      // Add remaining countries from negativeCounts
      for (const country in negativeCounts) {
        countsSheetData.push({
          Country: country,
          'Positive Count': 0,
          'Negative Count': negativeCounts[country],
        });
      }
      
      // Create a sheet for the sentiment data
      const sentimentSheetData = excelData.predictions.map((prediction) => ({
        Gender: prediction.gender,
        Age: prediction.age,
        Review: prediction.text,
        Country: prediction.country,
        Sentiment: prediction.sentiment,
        Reason: prediction.reasons ? prediction.reasons.join(', ') : '',
        Title: prediction.titles ? prediction.titles.join(', ') : '',
        Confidence: prediction.confidence,
      }));
      const countsSheet = XLSX.utils.json_to_sheet(countsSheetData);
      const sentimentSheet = XLSX.utils.json_to_sheet(sentimentSheetData);
  
      // Add both sheets to the same workbook
      XLSX.utils.book_append_sheet(wb, sentimentSheet, 'SentimentData');
      XLSX.utils.book_append_sheet(wb, countsSheet, 'CountryCounts');
  
      // Generate a Blob containing the workbook data
      const blob = XLSX.write(wb, { type: "array", bookType: 'xlsx' });
  
      const currentDateIST = new Date().toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
      });

      // Splitting the date and time parts
      const parts = currentDateIST.split(', ');
      const datePart = parts[0].replace(/ /g, '-');
      const timePart = parts[1].replace(/:/g, '.');
      
      // Removing spaces between time and AM/PM and adding a hyphen
      const timeAMPM = timePart.replace(/\s+/g, '') + '-';
      
      const uniqueFileName = `${datePart}-${timeAMPM}sentiment_analysis_file.xlsx`;
      
      // Replacing slashes with periods
      const uniqueFileNameProcessed = uniqueFileName.replace(/\//g, '.');
      
      
      // Construct the full storage path
      const storageRef = ref(storage, `data/${loggedUser}/${uniqueFileNameProcessed}`);
        try {
          // Upload the Blob to the Firebase Storage bucket
          await uploadBytes(storageRef, blob);
       
          // File has been successfully uploaded
          console.log('Excel file has been saved to Firebase Storage');
        } catch (error) {
          console.error('Error uploading Excel file to Firebase Storage', error);
          // Handle the error here
        }
      } else {
        // Handle the case where the user is not logged in
        console.log('User is not logged in. Please log in to upload the Excel file.');
      }
    }
  };
  
  const processCountryData = () => {
    const uniqueCountries = Array.from(new Set(countryData.map(data => data.country)));
  
    const processedData = uniqueCountries.map(country => {
      const positiveCount = positiveCounts[country] || 0;
      const negativeCount = negativeCounts[country] || 0;
      const total = positiveCount + negativeCount;
      const positivePercentage = total > 0 ? ((positiveCount / total) * 100).toFixed(2) : 0;
      const negativePercentage = total > 0 ? ((negativeCount / total) * 100).toFixed(2) : 0;
  
      return {
        country: country,
        positivePercentage: positivePercentage,
        negativePercentage: negativePercentage
      };
    });
  
    return processedData;
  };

  useEffect(() => {
    if (result.length === 0) {
      window.scrollTo(0, 0);
    } else {
      document.body.style.overflow = 'auto';
    }
  
    return () => {
      document.body.style.overflow = 'auto';
    };
  }, [result]);

  const renderCountryTable = () => {
    if (isLoading) {
      // Render loading indicator
    } else if (result.length > 1) {
      // Define color scales for positive and negative sentiment
      const positiveColorScale = scaleLinear()
        .domain([0, 1]) // Adjust the domain as needed
        .range(['#f7fcb9', '#addd8e', '#31a354']); // Light to dark green

      const negativeColorScale = scaleLinear()
        .domain([0, 1]) // Adjust the domain as needed
        .range(['#FFD3D3', '#FF0000']); // Light to dark red colors
      const processedCountryData = processCountryData();

      // Render the map with markers
      return (
      <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-start' }}>
          <div style={{ width: "100%"}}>
          <TableContainer component={Paper} style={{ border: '1px solid #ddd', borderRadius: '8px', width: "95%", marginLeft: "1%" }}>
          <Table sx={{ minWidth: "100%" }}>
            <TableHead style={{ backgroundColor: 'white' }}>
              <TableRow>
                <TableCell style={{ fontSize: '20px', fontWeight: 'bold', padding: '1%' }}>S.No</TableCell>
                <TableCell align="center" style={{ fontSize: '20px', fontWeight: 'bold', paddingLeft: '10%' }}>Country</TableCell>
                <TableCell align="center" style={{ fontSize: '20px', fontWeight: 'bold', paddingLeft: '15%' }}>Positive%</TableCell>
                <TableCell align="center" style={{ fontSize: '20px', fontWeight: 'bold', paddingLeft: '9%' }}>Negative%</TableCell>
              </TableRow>
            </TableHead>
          </Table>
        </TableContainer>
        <div style={{ maxHeight: '250px', overflowY: 'auto', width: '100%' }}>
          <TableContainer component={Paper} style={{ border: '1px solid #ddd', borderRadius: '8px', margin: '1%', width: "97.5%"}}>
            <Table>
              <TableBody>
                {processedCountryData.map((data, index) => {
                  const country = data.country;
                  const positiveCount = positiveCounts[country] || 0;
                  const negativeCount = negativeCounts[country] || 0;
                  const total = positiveCount + negativeCount;
                  const positivePercentage = total > 0 ? ((positiveCount / total) * 100).toFixed(2) : 0;
                  const negativePercentage = total > 0 ? ((negativeCount / total) * 100).toFixed(2) : 0;

                  return (
                    <TableRow style={{ backgroundColor: 'white'}} key={index}>
                      <TableCell align="center">{index+1}</TableCell>
                      <TableCell align="center">{data.country}</TableCell>
                      <TableCell align="center">{`${positivePercentage}%`}</TableCell>
                      <TableCell align="center">{`${negativePercentage}%`}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </div>
        </div> 
      </Box>
      );
    }
    return null;
  };

  const handleClearResults = () => {
    window.scrollTo(0, 0);
    setText("");
    setResult([]);
}

  const [isRecording, setIsRecording] = useState(false);
  const [recorder, setRecorder] = useState(null);
  const [recordedAudioFile, setRecordedAudioFile] = useState(null);
  
  const handleLiveAudioAnalyse = async (fileName) => {
    setError(null);
    setIsLoading(true);
    const loggedUser = localStorage.getItem('username');
  
    try {
      if (loggedUser) {
        const audioRef = ref(storage, `data/${loggedUser}/${fileName}`);
        const downloadURL = await getDownloadURL(audioRef);
        
        // Download the audio file
        const response = await fetch(downloadURL);
        const blob = await response.blob();
  
        // Create FormData and append the audio file
        const formData = new FormData();
        formData.append('audioFile', blob, fileName);
  
        // Send the FormData to your Flask server
        const analysisResponse = await axios.post('http://localhost:5000/audio_live_sentiment', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
  
        console.log(analysisResponse.data);
  
        // Process the response as needed
        setResult(
          analysisResponse.data.predictions.map((predictions) => ({
            text: predictions.text,
            sentiment: predictions.sentiment,
            reason: predictions.reasons,
            confidence: predictions.confidence,
            title_words: predictions.title_words,
          }))
        );
      }
    } catch (error) {
      console.error('Error fetching or analyzing audio:', error);
      setError('Error occurred while fetching or analyzing the audio file');
    } finally {
      setIsLoading(false);
      setAnalysisCompleted(true);
      setFile(null);
    };
  };
  
  
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
  
      const chunks = []; // Store chunks locally
  
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };
  
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: 'audio/wav' });
        setRecordedAudioFile(audioBlob);
  
        // Upload the recorded audio file to Firebase Storage
        uploadAudioToFirebase(audioBlob, 'my_audio.wav');
      };
  
      mediaRecorder.start();
      setIsRecording(true);
      setRecorder(mediaRecorder);
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };
  
  const stopRecording = () => {
    if (recorder) {
      recorder.stop();
      setIsRecording(false);
    }
  };
  
  const handleStartStopRecording = () => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  };
  
  const uploadAudioToFirebase = (audioBlob, filename) => {
    const loggedUser = localStorage.getItem('username');
    const currentDateIST = new Date().toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata',
    });
    
    // Splitting the date and time parts
    const parts = currentDateIST.split(', ');
    const datePart = parts[0].replace(/ /g, '-');
    const timePart = parts[1].replace(/:/g, '.');
          
    // Removing spaces between time and AM/PM and adding a hyphen
    const timeAMPM = timePart.replace(/\s+/g, '') + '-';
          
    const uniqueFileName = `${datePart}-${timeAMPM}audio_file.wav`;
    // Replacing slashes with periods
    const uniqueFileNameProcessed = uniqueFileName.replace(/\//g, '.');
    console.log("file name: ", uniqueFileNameProcessed)
    const storageRef = ref(storage, `data/${loggedUser}/${uniqueFileNameProcessed}`);    
    // Upload the audio file to the specified storage reference
    uploadBytes(storageRef, audioBlob)
      .then((snapshot) => {
        console.log('Audio file uploaded successfully:', snapshot);
        handleLiveAudioAnalyse(uniqueFileNameProcessed);
      })
      .catch((error) => {
        console.error('Error uploading audio file:', error);
      });
  };

    return (
        <>
        <div style={{ width: '100%', height: '100%', overflowY: 'auto' }}>
            <Box id="custombox">
                <h2 id='boxheader'>Sentiment Analysis</h2>
                <h4 id='boxheader2'>Type Your Text, Upload Dataset or Audio</h4>
                <div style={{ display: "flex", justifyContent: "center" }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                    <input style={{ width: "400px" }} id='boxsearch' placeholder='example: Jio Fiber Network is Fast' type="text" value={text} onChange={handleTextChange}/>
                    {result != null && result.length >= 1 ? (
                      <ClearIcon onClick={handleClearResults} />
                    ) : (
                    // <button onClick={handleStartStopRecording} style={{ color: isRecording ? 'red' : 'black' }}>
                    //   <KeyboardVoiceIcon />
                    // </button>
                    null
                    )}
                </div>
                </div>
                <div style={{ display: "flex", justifyContent: "center" }}>
                    <div style={{ display: "flex", flexDirection: "row", gap: 10 }}>
                        <Button id='btns' onClick={handleTextAnalyseClick} disabled={file !== null || isLoading} >Analyse</Button>
                        <label htmlFor="file" style={{ display: 'flex', alignItems: 'center' }}>
                          <Button id='btns' component="span" variant="contained" disabled={isLoading}>
                            Upload
                          </Button>
                          <input type="file" id="file" name="file" style={{ display: 'none' }} onChange={handleFileChange} accept=".xlsx, .csv, .mp3, .wav"/>
                        </label>
                    </div>
                </div>
                <h5 style={{ textAlign: "center", paddingTop: "30px", fontWeight: "400" }} >(For File Upload only xlsx, csv, wav, mp3 format files are accepted)</h5>
                <h3 style={{ textAlign: "center",  fontWeight: "400",marginTop: '5%', color: 'red' }} >{error}</h3>
            </Box>
            <div id='tex' style={{ position: "absolute", top: "550px" }}>
              <img style={{ width: "1700px" }} src={BGimg} alt="img" />
            </div>
            {error && (
              <Box style={{ marginTop: '-5%' }}>
                <Lottie
                  animationData={errorAnimationData}
                  loop={true}
                  style={{
                    width: '400px', // Set your desired width
                    height: '200px', // Set your desired height
                  }}/>
              </Box>
            )}
            {isLoading ? ( 
              <div style={{ height: '150px' }}>
                <Lottie animationData={animationData} loop={true} />
              </div>
            ) : result != null && result.length > 1 ? (
            <div id="insightPDF" style={{ width: '100%' }}>
              <Box ref={resultBoxRef1}>
                <div style={{ backgroundColor: '#ddd'}}>
                <Grid container spacing={2} style={{ margin: 0, marginTop: '28%' }} id="mainGrid">
                  <Grid xs={6} md={7} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF'}}>
                  <div>
                  <div>
                    <h2 align='center'>Sentiment Data</h2>
                    <div style={{ padding: '1% 1%' }}>
                    <TableContainer sx={{ width: '865px', backgroundColor: 'white', border: '1px solid #ddd', borderRadius: '8px', marginTop: '10px', marginLeft: '0.5%' }}>
                      <Table sx={{ minWidth: 10 }} aria-label="caption table">
                        <TableHead>
                          <TableRow style={{ backgroundColor: 'white' }}>
                            <TableCell align="center" style={{ fontSize: '18px', fontWeight: 'bold', width: '11%', paddingLeft: '18%' }}>
                              Review
                            </TableCell>
                            <TableCell align="center" style={{ fontSize: '18px', fontWeight: 'bold', width: '10%', paddingLeft: '15%' }}>
                              Sentiment
                              <IconButton
                                onClick={(event) => setSentimentMenuAnchor(event.currentTarget)}
                              >
                              <ArrowDropDownIcon style={{ position: 'absolute' }}/>
                              </IconButton>
                              <Menu
                                anchorEl={sentimentMenuAnchor}
                                open={Boolean(sentimentMenuAnchor)}
                                onClose={() => setSentimentMenuAnchor(null)}
                              >
                                <MenuItem onClick={() => setSentimentFilter('Both')}>Both</MenuItem>
                                <MenuItem onClick={() => setSentimentFilter('Positive')}>Positive</MenuItem>
                                <MenuItem onClick={() => setSentimentFilter('Negative')}>Negative</MenuItem>
                              </Menu>
                            </TableCell>
                            <TableCell align="center" style={{ fontSize: '18px', fontWeight: 'bold', width: '5%' }}>
                              Topic
                            </TableCell>
                            <TableCell align="center" style={{ fontSize: '18px', fontWeight: 'bold', width: '1%' }}>
                              Confidence
                              <IconButton
                                onClick={(event) => setConfidenceMenuAnchor(event.currentTarget)}
                              >
                                <ArrowDropDownIcon style={{ position: 'absolute' }}/>
                              </IconButton>
                              <Menu
                                anchorEl={confidenceMenuAnchor}
                                open={Boolean(confidenceMenuAnchor)}
                                onClose={() => setConfidenceMenuAnchor(null)}
                              >
                                <MenuItem onClick={() => setConfidenceFilter('All')}>All</MenuItem>
                                <MenuItem onClick={() => setConfidenceFilter('Below 40')}>Below 40</MenuItem>
                                <MenuItem onClick={() => setConfidenceFilter('40-50')}>40-60</MenuItem>
                                <MenuItem onClick={() => setConfidenceFilter('51-60')}>51-60</MenuItem>
                                <MenuItem onClick={() => setConfidenceFilter('61-70')}>61-70</MenuItem>
                                <MenuItem onClick={() => setConfidenceFilter('71-80')}>71-80</MenuItem>
                                <MenuItem onClick={() => setConfidenceFilter('81-90')}>81-90</MenuItem>
                                <MenuItem onClick={() => setConfidenceFilter('91-100')}>91-100</MenuItem>
                              </Menu>
                            </TableCell>
                          </TableRow>
                        </TableHead>
                      </Table>
                    </TableContainer>
                    <div style={{ maxHeight: '670px', overflowY: 'auto', marginLeft: '0.5%' }}>
                      <TableContainer sx={{ width: '99.8%', backgroundColor: 'white', border: '1px solid #ddd', borderRadius: '8px', marginTop: '10px' }}>
                        <Table>
                          <TableBody>
                            {renderRows()}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </div>
                    </div>
                  </div>
                </div>
                  </Grid>
                  <Grid xs={6} md={5} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF'}}>
                  <Grid item xs={12} md={12} style={{height: '50'}}>
                    <div>
                      <h2 align= 'center'>Topic Distribution</h2>
                      <div style={{ padding: '2% 2%' }}>
                        <Chart
                            chartType="BarChart"
                            width="100%"
                            height="380px"
                            data={titleChartData}
                            options={{
                              chartArea: { width: '70%', height: '75%' },
                              hAxis: { title: 'Count' },
                              vAxis: { title: 'Title', minValue: 0 },
                              legend: 'none',
                            }}
                          /> 
                      </div> 
                    </div>
                  </Grid>
                  <Grid item xs={12} md={12} style={{ }}>
                  <div style={{ width: '100%', height: '350px', display: 'flex', alignItems: 'center' }}>
                    <div style={{ width: '100%', marginTop: '-20%', position: 'absolute' }}>
                      <h2 style={{ paddingLeft: '11%' }}>Overall Sentiment Analysis</h2>
                    </div>
                    <div style={{ width: '50%' }}>
                      <div style={{  marginTop: '15%', display: 'flex' }}>
                        <div style={{ width: '100%' }}>
                        <Chart
                          chartType="PieChart"
                          data={overAllPosChartData}
                          options={{
                            chartArea: { width: '100%', height: '100%' }, 
                            colors: ['#39D069', '#B6AEAE'],
                            legend: { position: 'none' },
                            pieHole: 0.6,
                            backgroundColor: 'none',
                          }}
                          graph_id="PieChart1"
                          width={'100%'} 
                          height={'250px'}
                          legend_toggle
                        />
                        </div>
                        <div style={{ marginLeft: '-65%', width: '30%' }}>
                        <Chart
                            chartType="PieChart"
                            data={overAllNegChartData}
                            options={{
                              chartArea: { width: '100%', height: '100%' }, 
                              colors: ['#FA2D2D', '#B6AEAE'],
                              legend: { position: 'none' },
                              pieHole: 0.2,
                              backgroundColor: 'none',
                              pieStartAngle: pieStartAngle,
                              dataLabels: {
                                color: 'red', // Set the color of the data labels
                                textStyle: {
                                  fontName: 'Arial', // Set the font family
                                  fontSize: 14, // Set the font size
                                },
                              },
                            }}
                            graph_id="PieChart2"
                            width={'100%'} 
                            height={'250px'}
                            legend_toggle
                          />
                        </div>
                      </div>
                    </div>
                    <div style={{ justifyContent: 'flex-start', height: "70%", width: '100%', marginTop: '20%' }}>
                      <h4 style={{ fontSize: '13px' }}>Positive Sentiments : {posSenWidth} <br></br> Positive Count: {totalPositiveCounts}</h4>
                      <div style={{  width:'80%', border: '4px solid green', height: '20px', borderRadius: '35px', margin: '2%'}}>
                      <div style={{ width: posSenWidth  , height: '100%', backgroundColor: '#3DE363', borderRadius: '20px' }}></div>  
                      </div>
                      <h4 style={{ fontSize: '13px' }}>Negative Sentiments : {negSenWidth} <br></br> Negative Count: {totalNegativeCounts}</h4>
                      <div style={{  width:'80%', border: '4px solid red', height: '20px', borderRadius: '35px', margin: '2%'}}>
                      <div style={{ width: negSenWidth  , height: '100%', backgroundColor: '#FF6666', borderRadius: '20px' }}></div>  
                      </div>
                      <div><h4 style={{ fontSize: '20px' }}> Total Sentiments : {totalPositiveCounts + totalNegativeCounts}</h4></div>
                    </div>
                  </div>
                  </Grid>
                  </Grid>
                  {ageChartData.some(dataPoint => dataPoint[1] > 0 || dataPoint[2] > 0) && (
                  <Grid xs={6} md={4} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF'}}>
                  <div>
                    <div>
                      <div>
                      <h2 align='center'>Age Distribution</h2>
                        <div>
                        <Chart
                          width={'100%'}
                          height={'380px'}
                          chartType="BarChart"
                          loader={<div>Loading Chart</div>}
                          data={ageChartData}
                          options={{
                            chartArea: { width: '80%', height: '70%' },
                            chart: {
                              title: 'Age Distribution',
                              subtitle: 'Positive and Negative Sentiments by Age Group',
                            },
                            orientation: 'horizontal',
                            vAxis: {
                              title: 'Count',
                              minValue: 0,
                            },
                            hAxis: {
                              title: 'Age Group',
                            },
                            legend: { position: 'bottom' },
                          }}
                        />
                        </div>
                        </div>
                    </div>
                </div>
              </Grid>
                )}
                {genderChartData.length > 0 && (
                <Grid xs={6} md={4} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF'}}>
                <div>
                  <h2 align='center'>Gender Ratio</h2>
                    <div>
                    <Chart
                          width={'100%'}
                          height={'400px'}
                          chartType="PieChart"
                          loader={<div>Loading Chart</div>}
                          data={[
                            ['Gender', 'Count'],
                            ...genderChartData.map((data) => [data.label, data.value]),
                          ]}
                          options={{
                            chartArea: { width: '80%', height: '80%' },
                            pieHole: 1.0, // Adjust this value for the inner radius
                            colors: genderChartData.map((data) => data.style.fill), // Customize colors based on style.fill
                            legend: { position: 'bottom' },
                          }}
                    />
                    </div>
                </div>
                </Grid>
                )}
                {countryPlotData.length >= 1 && (
                <Grid xs={6} md={4} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF'}}>
                      <h2 align='center'>Country Distribution</h2>
                      <div style={{ backgroundColor: '#ddd'}}>
                        <div>
                        <Chart
                          chartType="BarChart"
                          data={countryPlotData}
                          options={{
                            orientation: 'horizontal',
                            chartArea: { width: '80%', height: '75%' },
                            legend: { position: 'none' },
                            hAxis: {
                              title: 'Country',
                            },
                            vAxis: {
                              title: 'Count',
                            },
                          }}
                          width="100%"
                          height="380px"
                        />
                      </div> 
                      </div>
                </Grid>
                )}
                {countryPlotData.length >= 1 && (
                <Grid xs={6} md={7} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF'}}>
                      <div>
                      <h2 align='center'>Regions wise Analysis</h2>
                      <Chart
                          width={'100%'}
                          height={'655px'}
                          chartType="GeoChart"
                          data={geoChartData}
                          options={{
                            colorAxis: {
                              minValue: 0,
                              colors: ['#ff0000', '#008000'],
                              strokeWidth: '50px',
                            },
                            backgroundColor: '#539fc3',
                          }}
                          loader={<div>Loading Chart</div>}
                        />
                      </div>
                </Grid>
                )}
              <Grid xs={6} md={5} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF' }}>
                  <div>
                  <h2 align='center'>Word Cloud</h2>
                    <div style={{ width: '96%', height: '100%'}}>
                    <canvas
                      ref={wordCloudCanvasRef}
                      width={800}
                      height={400}
                      style={{ width: '100%', height: '100%' }}
                    />
                    </div>
                    {countryPlotData.length >= 1 && (
                      <div>
                        <h2 align='center'>Country Data</h2>
                        {renderCountryTable()}
                      </div>
                    )}    
                  </div> 
              </Grid>
                </Grid>
                {insight && (
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', paddingTop:  '1%', paddingBottom: '2%'}}>
                  <div style={{ backgroundColor: 'white', width: '60%', paddingBottom: '3%' }}>
                    <div >
                    <Box >
                      <h2 align= 'center' style={{ fontSize: '35px', textDecoration: 'underline'}}>Overall Insights</h2>
                      <h2 align= 'center' style={{ fontSize: '20px', paddingTop: '3%' }}>
                        The Net Sentiment score indicating a {insight.net_sentiment_score.toFixed(2)}% difference between Positive and Negative,
                        highlights a prevailing {insight.prevailing_sentiment} sentiment, influenced by the factors such as{' '}
                        {insight.prevailing_sentiment === 'positive'
                          ? insight.top_titles_for_positive.join(', ')
                          : insight.top_titles_for_negative.join(', ')}
                        .
                      </h2>   
                      <Grid container spacing={2} style={{margin: 0, padding: '1.2%', height: '630px', paddingTop: '5%'}} id="mainGrid"> 
                      <Grid xs={4} md={5} style={{  backgroundColor: '#FFFFFF', marginLeft: '5%'}}>
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'center' }}>
                        <h2 style={{ color: 'red', fontWeight: 'bold', fontSize: '30px' }}>Negative Sentiment</h2>
                        <SentimentVeryDissatisfiedIcon style={{ fontSize: '50px', color: 'red' }} />
                        </div>
                        <div style={{ marginLeft: '2%' }}>
                        <Chart
                          width={'100%'}
                          height={'300px'}
                          chartType="PieChart"
                          loader={<div>Loading Chart</div>}
                          data={negChart}
                          options={{
                            title: '',
                            chartArea: { width: '90%', height: '80%' },
                            pieHole: 0.6, // Adjust this value to control the size of the hole (0.6 means 60% filled, 40% empty)
                            colors: ['#db1c1c', '#ddd'], // Adjust colors as needed
                            legend: 'none'
                          }}
                        />
                        <div style={{ display: 'flex', justifyContent: 'center' }}>  
                          <ul>
                            <li>
                            <h4 >Concerns includes {insight.top_reasons_for_negative.join(', ')}</h4> 
                            </li>
                          </ul>
                        </div>
                        <div style={{ paddingTop: '10%' }}>
                          <h2 align='center' style={{ fontWeight: 'bold', fontSize: '32px'}}>
                            Top Reasons for Negative Sentiment
                          </h2>
                          <div style={{ marginTop: '5%' }}>
                            <div style={{ display: 'flex', justifyContent: 'center' }}>
                              {insight.top_titles_for_negative.map((title, index) => (
                                <div
                                key={index}
                                style={{
                                  backgroundColor: '#db1c1c', /* Background color for the boxes */
                                  color: '#fff', /* Text color */
                                  padding: '5px 8px', /* Adjust padding as needed */
                                  margin: '5px', /* Adjust margin as needed */
                                  borderRadius: '20px', /* Curved borders */
                                }}
                              >
                                  {title}
                                </div>
                              ))}
                            </div>
                          </div> 
                        </div>
                        </div>
                      </div>
                      </Grid>   
                      <Grid xs={4} md={5} style={{ backgroundColor: '#FFFFFF', marginLeft: '5%'}}>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'center' }}>
                          <h2 style={{ color: 'green', fontWeight: 'bold', fontSize: '30px' }}>Positive Sentiment</h2>
                          <SentimentSatisfiedAltIcon style={{ verticalAlign: 'middle', fontSize: '50px', color: 'green' }} />
                          </div>
                          <div>
                          <Chart
                            width={'100%'}
                            height={'300px'}
                            chartType="PieChart"
                            loader={<div>Loading Chart</div>}
                            data={posChart}
                            options={{
                              title: '',
                              chartArea: { width: '90%', height: '80%' },
                              pieHole: 0.6, // Adjust this value to control the size of the hole (0.6 means 60% filled, 40% empty)
                              colors: ['#2fae20', '#ddd'], // Adjust colors as needed
                              legend: 'none',
                              pieStartAngle: pieEndAngle
                            }}
                          />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'center' }}>  
                          <ul>
                            <li style={{ marginLeft: '15%' }}>
                            <h4>Customers appreciate {insight.top_reasons_for_positive.join(', ')}</h4> 
                            </li>
                          </ul>
                        </div>
                        <div style={{ paddingTop: '5%', marginLeft: '12%' }}>
                          <h2 align='center' style={{ fontWeight: 'bold', fontSize: '32px'}}>
                            Top Reasons for Positive Sentiment
                          </h2>
                          <div style={{ marginTop: '5%' }}>
                            <div style={{ display: 'flex', justifyContent: 'center' }}>
                              {insight.top_titles_for_positive.map((title, index) => (
                                <div
                                key={index}
                                style={{
                                  backgroundColor: '#2fae20', /* Background color for the boxes */
                                  color: '#fff', /* Text color */
                                  padding: '5px 3px', /* Adjust padding as needed */
                                  margin: '5px', /* Adjust margin as needed */
                                  borderRadius: '20px', /* Curved borders */
                                }}
                              >
                                  {title}
                                </div>
                              ))}
                            </div>
                          </div> 
                        </div>
                        </div>
                      </Grid>
                      </Grid>    
                      {
                        insight.recommendations.length >=1 && (
                          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginLeft: '1%', marginTop: '4%' }}>
                        <div style={{ width: '80%', backgroundColor: '#EBE5E5', borderRadius: '15px' }}>
                          <h2  style={{ fontSize: '30px', fontWeight: 'bold', marginLeft: '28%' }} >Recommended Actions</h2>
                          {insight.recommendations.map((recommendation, index) => (
                            <div style={{ fontSize: '25px', marginLeft: '4%' }} key={index}><CheckCircleOutlineIcon style={{ color: 'green'}} />    {recommendation}</div>
                          ))}
                        </div>
                      </div>
                        )
                      }
                      </Box>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '5%' }}>
                      <Button id="btns" variant="contained" sx={{ height: '50px', backgroundColor: '#E0E4FD', color: 'black', marginRight: '5%' }} onClick={handleExcelDownloadClick}>
                        Download Excel File
                      </Button>
                      <Button
                      id="btns"
                        variant="contained"
                        sx={{ height: '50px', backgroundColor: '#E0E4FD', color: 'black' }}
                        onClick={handleSaveAsPdf}
                      >
                        Save as PDF
                      </Button>
                    </div>
                  </div>
                </div>
                )}
                </div> 
              </Box> 
            </div>
              ) : 
             ( result.length === 1 && analysisCompleted &&  (     
              <Box id="custombox1" ref={resultBoxRef} style={{ borderColor: result[0].sentiment === 'Positive' ? '#B9EACB' : 'red', borderWidth: '2px', borderStyle: 'solid' }}>
              <div>
              <h2 style={{  }}>{highlightWordsText(result[0].text, result[0].reason, result[0].sentiment, result[0].title_words)}</h2><br></br>
              {result[0].titles && Array.isArray(result[0].titles) ? (
                result[0].titles.map((title, index) => (
                  <div
                    style={{
                      display: 'inline-block',
                      border: '2px solid',
                      padding: '5px', // Adjust the padding as needed
                      margin: '5px',  // Adjust the margin as needed
                      borderRadius: '35px',
                      color: "#04BADE",
                    }}
                  >
                    <h3 align="center" class = "title" style={{ fontWeight: 'normal' }}>{title}</h3>
                  </div>
                ))
              ) : null}
              <h3 id="resText" style={{ fontWeight: 'normal', color: '#404A86', marginTop: '4%' }} align= 'center'>The Given Sentence is {result[0].sentiment} with confidence of {result[0].confidence}%</h3>
              </div>
              </Box>
              ))}
            </div>
            <Box style={{ marginTop: 
              result.length === 1 ? '20%' :
              result.length > 1 ? '0%' :
              '25%'
            }}>
              <Footer />
            </Box>
        </>
    )
}

export default Sentiment
