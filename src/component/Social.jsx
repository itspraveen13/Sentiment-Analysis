import React, { useState, useEffect, useRef } from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import axios from 'axios';
import { TypeAnimation } from 'react-type-animation';
import * as XLSX from 'xlsx';
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { storage } from '../assets/js/config.js';
import { Chart } from "react-google-charts";
import SentimentSatisfiedAltIcon from '@mui/icons-material/SentimentSatisfiedAlt';
import SentimentVeryDissatisfiedIcon from '@mui/icons-material/SentimentVeryDissatisfied';
import Grid from '@mui/material/Unstable_Grid2'; 
import WordCloud from 'wordcloud';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import { Bar } from 'react-chartjs-2';
import {
  Box,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Menu, 
  MenuItem, 
  IconButton,
} from '@mui/material';
import ClearIcon from '@mui/icons-material/Clear';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown'; // Import the arrow icon
import Lottie from 'lottie-react';
import animationData from '../assets/js/Animation.json';
import BGimg from '../assets/img/BG_IMG.png'
import html2canvas from 'html2canvas';
import { saveAs } from 'file-saver';
import jsPDF from 'jspdf';
import Footer from '../component/Footer.jsx'

const Social = () => {
    const [startDate, setStartDate] = useState(new Date());
    const [endDate, setEndDate] = useState(new Date());
    const [keyword, setKeyword] = useState('');
    const [isLoading, setIsLoading] = useState(false); // Loading state
    const [showOutput, setShowOutput] = useState(false); // State to control output visibility
    const [excelData, setExcelData] = useState(null);
    const today = new Date();
    today.setHours(0, 0, 0, 0); 
    const [insight, setInsights] = useState(null);
    const [posChart, setPosChart] = useState('null');
    const [negChart, setNegChart] = useState('null');
    const [pieStartAngle, setPieStartAngle] = useState('null');
    const [wordCloudData, setWordCloudData] = useState({});
    const [titleChartData, setTitleChartData] = useState([]);
    const [pieEndAngle, setPieEndAngle] = useState('null');
    const [result, setResult] = useState([]);
    const [overAllPosChartData, setOverAllPosChartData] = useState([]);
    const [overAllNegChartData, setOverAllNegChartData] = useState([]);
    const [posSenWidth, setPosSenWidth] = useState('null');
    const [negSenWidth, setNegSenWidth] = useState('null');
    const [totalPositiveCounts, setTotalPositiveCounts] = useState(0);
    const [totalNegativeCounts, setTotalNegativeCounts] = useState(0);
    const [redditPositive, setRedditPositive] = useState('null');
    const [redditNegative, setRedditNegative] = useState('null');
    const [youtubePositive, setYoutubePositive] = useState('null');
    const [youtubeNegative, setYoutubeNegative] = useState('null');
    const [redditTitleChartData, setRedditTitleChartData] = useState([]);
    const [youtubeTitleChartData, setYoutubeTitleChartData] = useState([]);
    const [selectedChart, setSelectedChart] = useState('youtube'); // Default to YouTube chart
    const [youtubePieChartData, setYoutubePieChartData] = useState([]);
    const [redditPieChartData, setRedditPieChartData] = useState([]);
    const [youtubeResult, setYoutubeResult] = useState([]);
    const [redditResult, setRedditResult] = useState([]);
    const [textInput, setTextInput] = useState('');
    const [sentimentFilter, setSentimentFilter] = useState('Both'); // 'Positive', 'Negative', or 'Both'
    const [confidenceFilter, setConfidenceFilter] = useState('All'); // '40-60', '61-80', '81-100', or 'All'
    const [sentimentMenuAnchor, setSentimentMenuAnchor] = useState(null);
    const [confidenceMenuAnchor, setConfidenceMenuAnchor] = useState(null);
    const [count, setCount] = useState(100);
    const [error, setError] = useState("");
    const resultBoxRef = useRef(null);
    const wordCloudCanvasRef = useRef(null);
    const [ stackedTitleChartData,  setStackedTitleChartData] = useState([]);
    const [groupedBarChartData, setGroupedBarChartData] = useState([]);

    const handleScrollIntoView = () => {
      if (resultBoxRef.current) {
        resultBoxRef.current.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }
    };
  
    useEffect(() => {
      // Scroll into view when the component mounts and conditions are met
      if (result) {
        handleScrollIntoView();
      }
    }, [result]);

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

    const toggleChart = (chart) => {
      setSelectedChart(chart);
    };
  
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
          saveAs(pdfBlob, 'Social.pdf');
        });
    };

    const renderRows = () => {
      return filteredResults.map((res, index) => (
        <TableRow key={index}>
          <TableCell style={{ fontSize: '18px', width: '15%' }}>
            {highlightWords(res.text, res.reason, res.sentiment)}
          </TableCell>
          <TableCell align="center" style={{ fontSize: '16px', width: '1%' }}>
            {res.sentiment}
          </TableCell>
          <TableCell align="center" style={{ fontSize: '15px', width: '5%' }}>
            {res.title.length > 1 ? res.title.join(', ') : res.title[0]}
            {res.title.length === 0 ? 'None' : ''}
          </TableCell>
          <TableCell align="center" style={{ fontSize: '16px', width: '1%' }}>
            {res.confidence}
          </TableCell>
        </TableRow>
      ));
    };

    const fetchPredictions = () => {
      const keyword = textInput;
      setResult([]);
      axios
        .post('http://localhost:5000/social', {
          start_date: startDate.toISOString().slice(0, 10),
          end_date: endDate.toISOString().slice(0, 10),
          keyword,
          count,
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
            }))
          );
          setInsights(response.data.insight);
          setShowOutput(true); // Set showOutput to true when predictions are received
          setIsLoading(false); // Set loading state to false after receiving predictions
          setExcelData(response.data);
          setRedditResult(response.data.reddit);
          setYoutubeResult(response.data.youtube);
          const posSenCount = response.data.insight.positive_sentiment_count;
          const negSenCount = response.data.insight.negative_sentiment_count;
          const totalSen = posSenCount + negSenCount;
          const positivePercentage = (posSenCount / totalSen) * 100;
          const negativePercentage = (negSenCount / totalSen) * 100;
          const posEmpty = 100 - positivePercentage;
          const negEmpty = 100 - negativePercentage;
          const startAngle = 0; 
          const endAngleFirstChart = startAngle + (positivePercentage / 100) * 360; 
          const engAngleSecondChart = startAngle + (negativePercentage / 100) * 360;

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
  
          const ytPos = response.data.youtube_positive;
          const ytNeg = response.data.youtube_negative;
          const redditPos = response.data.reddit_positive;
          const redditNeg = response.data.reddit_negative;
  
          const youtubeTitleCounts = {};
          response.data.youtube.forEach((prediction) => {
              prediction.titles.forEach((title) => {
                  if (youtubeTitleCounts[title]) {
                    youtubeTitleCounts[title]++;
                  } else {
                    youtubeTitleCounts[title] = 1;
                  }
              });
          });
          
          const youtubeTitleChartData = [['Title', 'Count', { role: 'style' }]];
          Object.keys(youtubeTitleCounts).forEach((title) => {
            youtubeTitleChartData.push([title, youtubeTitleCounts[title], getRandomColor()]);
          });
          
          const redditTitleCounts = {};
          response.data.reddit.forEach((prediction) => {
              prediction.titles.forEach((title) => {
                  if (redditTitleCounts[title]) {
                    redditTitleCounts[title]++;
                  } else {
                    redditTitleCounts[title] = 1;
                  }
              });
          });
          
          const redditTitleChartData = [['Title', 'Count', { role: 'style' }]];
          Object.keys(redditTitleCounts).forEach((title) => {
            redditTitleChartData.push([title, redditTitleCounts[title], getRandomColor()]);
          });
  

          const youtubeTitleChartData1 = [['Title', 'YouTube', 'Reddit']];
          const allTitles = [...new Set([...Object.keys(youtubeTitleCounts), ...Object.keys(redditTitleCounts)])];
      
          allTitles.forEach((title) => {
            const youtubeCount = youtubeTitleCounts[title] || 0;
            const redditCount = redditTitleCounts[title] || 0;
            youtubeTitleChartData1.push([title, youtubeCount, redditCount]);
          });
      
          setStackedTitleChartData(youtubeTitleChartData1);

          const redditPositiveCount = response.data.reddit_positive;
          const redditNegativeCount = response.data.reddit_negative;
          const youtubePositiveCount = response.data.youtube_positive;
          const youtubeNegativeCount = response.data.youtube_negative;
          
          const chartData = [
            ['Sentiment', 'YouTube', 'Reddit'],
            ['Positive', redditPositiveCount, youtubePositiveCount],
            ['Negative', redditNegativeCount, youtubeNegativeCount],
          ];
      
          setGroupedBarChartData(chartData);
          setYoutubeTitleChartData(youtubeTitleChartData);
          setRedditTitleChartData(redditTitleChartData);
          setYoutubePositive(ytPos);
          setYoutubeNegative(ytNeg);
          setRedditPositive(redditPos);
          setRedditNegative(redditNeg);
          setWordCloudData(wordCloudData);
          setTotalPositiveCounts(posSenCount);
          setTotalNegativeCounts(negSenCount);
          setPosSenWidth(positivePercentage.toFixed(2) + '%');
          setNegSenWidth(negativePercentage.toFixed(2) + '%');
          setOverAllPosChartData(positiveCircle);
          setOverAllNegChartData(negativeCircle);
          setTitleChartData(titleChartData);
          setPosChart(positiveCircle);
          setNegChart(negativeCircle);
          setPieStartAngle(endAngleFirstChart);
          setPieEndAngle(engAngleSecondChart);
        })
        .catch((error) => {
          console.log(error);
          setError('Error occurred while analyzing the text');
          setIsLoading(false); // Set loading state to false if an error occurs
        });
    };
    
    
    function getRandomColor() {
      const letters = '0123456789ABCDEF';
      let color = '#';
      for (let i = 0; i < 6; i++) {
          color += letters[Math.floor(Math.random() * 16)];
      }
      return color;
  }
  
    useEffect(() => {
      if (excelData) {
        saveResultsToFirebase();
      }
    }, [excelData]);
  
    const saveResultsToExcel = () => {
      // Create a new Excel workbook
      const workbook = XLSX.utils.book_new();
  
      // Convert the analysis results into an array of rows
      const data = result.map((prediction) => ({
        Text: prediction.text,
        Sentiment: prediction.sentiment,
        Reasons: prediction.reason ? prediction.reason.join(', ') : '',
        Titles: prediction.title ? prediction.title.join(', ') : '',
        Confidence: prediction.confidence,
      }));
  
      const redditData = redditResult.map((prediction) => ({
        Text: prediction.text,
        Sentiment: prediction.sentiment,
        Reasons: prediction.reasons ? prediction.reasons.join(', ') : '',
        Titles: prediction.titles ? prediction.titles.join(', ') : '',
        Confidence: prediction.confidence,
      }));
  
      const youtubeData = youtubeResult.map((prediction) => ({
        Text: prediction.text,
        Sentiment: prediction.sentiment,
        Reasons: prediction.reasons ? prediction.reasons.join(', ') : '',
        Titles: prediction.titles ? prediction.titles.join(', ') : '',
        Confidence: prediction.confidence,
      }));
      // Create a worksheet and add the data
      const worksheet = XLSX.utils.json_to_sheet(data);
      const redditWorksheet = XLSX.utils.json_to_sheet(redditData);
      const youtubeWorksheet = XLSX.utils.json_to_sheet(youtubeData);
    
      const currentDateIST = new Date().toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
      });
      
      // Splitting the date and time parts
      const parts = currentDateIST.split(', ');
      const datePart = parts[0].replace(/ /g, '-');
      const timePart = parts[1].replace(/:/g, '.');
            
      // Removing spaces between time and AM/PM and adding a hyphen
      const timeAMPM = timePart.replace(/\s+/g, '') + '-';
            
      const uniqueFileName = `${datePart}-${timeAMPM}social_media_file.xlsx`;
            
      // Replacing slashes with periods
      const uniqueFileNameProcessed = uniqueFileName.replace(/\//g, '.');
           
      // Add both sheets to the same workbook
      XLSX.utils.book_append_sheet(workbook, worksheet, 'SocialMediaData');
      XLSX.utils.book_append_sheet(workbook, redditWorksheet, 'RedditData');
      XLSX.utils.book_append_sheet(workbook, youtubeWorksheet, 'YouTubeData');  
      XLSX.writeFile(workbook, uniqueFileNameProcessed);
    };
  
    const saveResultsToFirebase = async () => {
      const loggedUser = localStorage.getItem('username');
      if (loggedUser) {
      // Create a new Excel workbook
      const workbook = XLSX.utils.book_new();
  
      // Convert the analysis results into an array of rows
      // Convert the analysis results into an array of rows
      const data = result.map((prediction) => ({
        Text: prediction.text,
        Sentiment: prediction.sentiment,
        Reasons: prediction.reason ? prediction.reason.join(', ') : '',
        Titles: prediction.title ? prediction.title.join(', ') : '',
        Confidence: prediction.confidence,
      }));
  
      const redditData = redditResult.map((prediction) => ({
        Text: prediction.text,
        Sentiment: prediction.sentiment,
        Reasons: prediction.reasons ? prediction.reasons.join(', ') : '',
        Titles: prediction.titles ? prediction.titles.join(', ') : '',
        Confidence: prediction.confidence,
      }));
  
      const youtubeData = youtubeResult.map((prediction) => ({
        Text: prediction.text,
        Sentiment: prediction.sentiment,
        Reasons: prediction.reasons ? prediction.reasons.join(', ') : '',
        Titles: prediction.titles ? prediction.titles.join(', ') : '',
        Confidence: prediction.confidence,
      }));
      // Create a worksheet and add the data
      const worksheet = XLSX.utils.json_to_sheet(data);
      const redditWorksheet = XLSX.utils.json_to_sheet(redditData);
      const youtubeWorksheet = XLSX.utils.json_to_sheet(youtubeData);
      XLSX.utils.book_append_sheet(workbook, worksheet, 'SocialMediaData');
      XLSX.utils.book_append_sheet(workbook, redditWorksheet, 'RedditData');
      XLSX.utils.book_append_sheet(workbook, youtubeWorksheet, 'YouTubeData');  
  
      const currentDateIST = new Date().toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
      });
      
      // Splitting the date and time parts
      const parts = currentDateIST.split(', ');
      const datePart = parts[0].replace(/ /g, '-');
      const timePart = parts[1].replace(/:/g, '.');
            
      // Removing spaces between time and AM/PM and adding a hyphen
      const timeAMPM = timePart.replace(/\s+/g, '') + '-';
            
      const uniqueFileName = `${datePart}-${timeAMPM}social_media_file.xlsx`;
            
      // Replacing slashes with periods
      const uniqueFileNameProcessed = uniqueFileName.replace(/\//g, '.');
      const blob = XLSX.write(workbook, { type: "array", bookType: 'xlsx' });
      
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
      }
    };
  
    const handleStartDateChange = (date) => {
      setStartDate(date);
      if (date >= endDate) {
        setEndDate(new Date(date.getTime() + 24 * 60 * 60 * 1000));
      }
    };
  
    const handleEndDateChange = (date) => {
      setEndDate(date);
      if (date <= startDate) {
        setStartDate(new Date(date.getTime() - 24 * 60 * 60 * 1000));
      }
    };
  
    useEffect(() => {
      // Ensure that the start date is always one day before the end date initially
      if (startDate >= endDate) {
        setEndDate(new Date(startDate.getTime() + 24 * 60 * 60 * 1000));
      }
    }, []);
  
    const handleSubmit = () => {
      setError("");
      setIsLoading(true); // Set loading state to true
      setShowOutput(false); // Reset showOutput to false before making the API call
      fetchPredictions(); // Fetch predictions from the Flask API
    };

    function highlightWords(text, reasons, sentiment) {
      // Split the text into words
      const words = text
        .split(/\s+/) // Split by spaces
        .map((word) => word.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, '')); // Remove punctuation marks
    
      // Create a function to check if a word is in the reasons list
      const isReasonWord = (word) => reasons.includes(word.toLowerCase());
    
      return words.map((word, index) => {
        // Check if the word is not empty and is a reason word, then apply styling accordingly
        const wordStyle =
          word.trim() !== '' &&
          isReasonWord(word) &&
          sentiment === 'Positive'
            ? { backgroundColor: '#B9EACB', color: '#3A805B', border: '2px solid #4F7F65',padding: '2px', borderRadius: '3px' }
            : word.trim() !== '' &&
              isReasonWord(word) &&
              sentiment === 'Negative'
            ? { backgroundColor: '#FCD7DE', border: '2px solid #C60943', padding: '2px',color: '#C60943', borderRadius: '3px' }
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

      const handleClearResults = () => {
        window.scrollTo(0, 0);
        setResult([]); 
        setShowOutput(false);
        setTextInput("");
    }
      
    return (
        <>
            <Box id="custombox2">
                <h2 id='boxheader'>Social Media Analysis</h2>
                <h4 style={{ marginTop: "90px" }} id='boxheader4'>Enter the Keyword</h4>
                <div style={{ display: 'flex' }}>
                <input
                    style={{ margin: '5px 0 10px 3px', width: '95%' }}
                    id="boxsearch"
                    placeholder="example: Jio Fiber"
                    type="text"
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                />
                {result != null && result.length >= 1 ? (
                  <ClearIcon style={{ marginTop: '1.3%' }} onClick={handleClearResults} />
                    ) : (
                      null
                    )}
                </div>
                <div style={{ display: "flex", justifyContent: "center" }}>
                </div>
                <div style={{ display: "flex", position: 'relative', marginTop: '2%' }}>
                  <div style={{ display: "flex", alignItems: "center", marginLeft: '2%' }}>
                      <h4 style={{ fontWeight: 500 }}>Maximum Records </h4>
                      <input
                          type="number"
                          value={count}
                          onChange={(e) => setCount(parseInt(e.target.value))}
                          style={{ width: "40px"}}
                      />
                  </div>
                  <div style={{ display: "flex", alignItems: "center", marginLeft: "20px" }}>
                      <h4 style={{ fontWeight: 500 }}>Start Date </h4>
                      <DatePicker selected={startDate} onChange={handleStartDateChange} maxDate={new Date() - 1} />
                  </div>
                  <div style={{ display: "flex", alignItems: "center", marginLeft: "20px" }}>
                      <h4 style={{ fontWeight: 500 }}>End Date </h4>
                      <DatePicker selected={endDate} onChange={handleEndDateChange} maxDate={new Date() - 1} />
                  </div>
              </div>
                <div style={{ display: "flex", justifyContent: "center" }}>
                <Button
                    style={{ marginTop: '20px' }}
                    onClick={handleSubmit}
                    id='btns'
                    disabled={!textInput || isLoading}> Analyse </Button>
                </div>
            </Box>
            <h4 align='center' style={{ color:'red', marginTop: '3%' }}>{error}</h4>
            <div id='tex' style={{ position: "absolute", top: "550px" }}>
                    <img style={{ width: "1700px" }} src={BGimg} alt="img" />
                  </div>
            {isLoading ? (
              <div style={{ height: '150px', marginTop: '5%' }}>
              <Lottie animationData={animationData} loop={true} />
            </div>
            ) : (
                showOutput && (
                  <div id = 'insightPDF'>
                    <Box style={{ backgroundColor: '#ddd', marginTop: '28%'}} ref={resultBoxRef}>
                    <Grid container spacing={2} style={{margin: 0, padding: '1.2%'}} id="mainGrid">
                    <Grid xs={6} md={7} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF'}}>
                    <div>
                      <div>
                        <h2 align='center'>Sentiment Data</h2>
                        <div style={{ padding: '1% 1%' }}>
                        <TableContainer sx={{ width: '815px', backgroundColor: 'white', border: '1px solid #ddd', borderRadius: '8px', marginTop: '10px', marginLeft: '0.5%' }}>
                          <Table sx={{ minWidth: 10 }} aria-label="caption table">
                            <TableHead>
                              <TableRow style={{ backgroundColor: 'white' }}>
                                <TableCell align="center" style={{ fontSize: '18px', fontWeight: 'bold', width: '25%', paddingLeft: '13%' }}>
                                  Review
                                </TableCell>
                                <TableCell align="center" style={{ fontSize: '18px', fontWeight: 'bold', width: '15%', paddingLeft: '9%' }}>
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
                                <TableCell align="center" style={{ fontSize: '18px', fontWeight: 'bold', width: '5%', paddingLeft: '5%' }}>
                                  Topic
                                </TableCell>
                                <TableCell align="center" style={{ fontSize: '18px', fontWeight: 'bold', width: '10%', paddingLeft: '5%' }}>
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
                    <Grid item xs={12} md={4} style={{height: '50'}}>
                        <div>
                        <h2 align='center'>Topic Distribution</h2>
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
                    </Grid>
                    <Grid item xs={12} md={6}>
                    <div style={{ width: '100%', height: '350px', display: 'flex', alignItems: 'center' }}>
                    <div style={{ width: '100%', marginTop: '-18%', position: 'absolute' }}>
                        <h2 style={{ paddingLeft: '10%' }}>Overall Sentiment Analysis</h2>
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
                      <h4 style={{ fontSize: '11px' }}>Positive Sentiments : {posSenWidth} | Positive Count: {totalPositiveCounts}</h4>
                      <div style={{  width:'90%', border: '4px solid green', height: '20px', borderRadius: '35px', margin: '2%'}}>
                      <div style={{ width: posSenWidth  , height: '100%', backgroundColor: '#3DE363', borderRadius: '20px' }}></div>  
                      </div>
                      <h4 style={{ fontSize: '11px' }}>Negative Sentiments : {negSenWidth} | Negative Count: {totalNegativeCounts}</h4>
                      <div style={{  width:'90%', border: '4px solid red', height: '20px', borderRadius: '35px', margin: '2%'}}>
                      <div style={{ width: negSenWidth  , height: '100%', backgroundColor: '#FF6666', borderRadius: '20px' }}></div>  
                      </div>
                      <div><h4 style={{ fontSize: '11px' }}>&#8226; Total Sentiments : {totalPositiveCounts + totalNegativeCounts}</h4></div>
                    </div>
                    </div>
                    </Grid>
                    </Grid>
                    <Grid xs={6} md={5} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF', height: '380px' }}>
                    <div>
                    <h2 align='center'>Word Cloud</h2>
                        <div style={{ width: '96%' }}>
                          <canvas
                            ref={wordCloudCanvasRef}
                            width={800}
                            height={400}
                            style={{ width: '100%', height: '100%' }}
                          />
                        </div>
                    </div>  
                    </Grid>
                    <Grid xs={6} md={7} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF', height: '380px' }}>
                    <Grid xs={6} md={7} >
                    <div style={{ display: 'flex', height: '100%' }}>
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                        <div>
                            <h2 style={{ fontSize: '15px', marginLeft: '13%' }} >Topic Distribution in Reddit and YouTube</h2>
                            <Chart
                                  width={'500px'}
                                  height={'320px'}
                                  chartType="BarChart"
                                  loader={<div>Loading Chart</div>}
                                  data={stackedTitleChartData}
                                  options={{
                                    isStacked: true,
                                    legend: { position: 'top' },
                                    chartArea: {
                                      backgroundColor: 'transparent', 
                                    },
                                  }}
                                  rootProps={{ 'data-testid': '1' }}
                            />
                        </div>
                        <div>
                        </div>
                        </div>
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '250px' }}>
                        <div>
                        <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <h2 style={{ marginLeft: '-10%', fontSize: '15px' }}>Sentiment Distribution in Reddit and YouTube</h2>
                            </div>
                        </div>
                        <div style={{ marginLeft: '-14%' }}>
                        <Chart
                                  width={'400px'}
                                  height={'320px'}
                                  chartType="ColumnChart"
                                  loader={<div>Loading Chart</div>}
                                  data={groupedBarChartData}
                                  options={{
                                    legend: { position: 'top' },
                                    bars: 'group',
                                    chartArea: {
                                      backgroundColor: 'transparent', 
                                    },
                                    hAxis: {
                                      title: 'Count',
                                    },
                                    vAxis: {
                                      title: 'Sentiment',
                                    },
                                  }}
                                  rootProps={{ 'data-testid': '1' }}
                                />
                          </div>
                          </div>
                        </div>
                    </div>
                    </Grid>
                    </Grid>
                    </Grid>
                    {insight && ( 
                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', paddingTop:  '1%', paddingBottom: '2%'}}>
                    <div style={{ backgroundColor: 'white', width: '60%', paddingBottom: '3%' }}>
                        <Box >
                        <h2 align= 'center' style={{ fontSize: '35px', textDecoration: 'underline'}}>Overall Insights</h2>
                        <h2 align= 'center' style={{ fontSize: '20px', paddingTop: '3%'}}>
                            The Net Sentiment score indicating a {insight.net_sentiment_score.toFixed(2)}% difference between Positive and Negative,
                            highlights a prevailing {insight.prevailing_sentiment} sentiment, influenced by the factors such as{' '}
                            {insight.prevailing_sentiment === 'positive'
                            ? insight.top_titles_for_positive.join(', ')
                            : insight.top_titles_for_negative.join(', ')}
                            .
                        </h2>   
                        <Grid container spacing={2} style={{margin: 0, padding: '1.2%', height: '630px', paddingTop: '5%'}} id="mainGrid"> 
                        <Grid xs={4} md={6} style={{  backgroundColor: '#FFFFFF'}}>
                        <div>
                            <div style={{ display: 'flex', justifyContent: 'center' }}>
                            <h2 style={{ color: 'red', fontWeight: 'bold', fontSize: '30px' }}>Negative Sentiment</h2>
                            <SentimentVeryDissatisfiedIcon style={{ fontSize: '50px', color: 'red' }} />
                            </div>
                            <div style={{ marginLeft: '2%'}}>
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
                                colors: ['#FA2D2D', '#ddd'], // Adjust colors as needed
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
                                    backgroundColor: '#FA2D2D', /* Background color for the boxes */
                                    color: '#fff', /* Text color */
                                    padding: '5px 10px', /* Adjust padding as needed */
                                    margin: '1px', /* Adjust margin as needed */
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
                        <Grid xs={4} md={6} style={{ backgroundColor: '#FFFFFF'}}>
                            <div>
                            <div style={{ display: 'flex', justifyContent: 'center' }}>
                            <h2 style={{ color: 'green', fontWeight: 'bold', fontSize: '30px' }}>Positive Sentiment</h2>
                            <SentimentSatisfiedAltIcon style={{ verticalAlign: 'middle', fontSize: '50px', color: 'green' }} />
                            </div>
                            <div style={{ marginLeft: '2%'}}>
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
                                colors: ['#39D069', '#ddd'], // Adjust colors as needed
                                legend: 'none',
                                pieStartAngle: pieEndAngle,
                                }}
                            />
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'center' }}>  
                            <ul>
                                <li>
                                <h4>Customer appreciates {insight.top_reasons_for_positive.join(', ')}</h4> 
                                </li>
                            </ul>
                            </div>
                            <div style={{ paddingTop: '4.5%' }}>
                            <h2 align='center' style={{ fontWeight: 'bold', fontSize: '32px'}}>
                                Top Reasons for Positive Sentiment
                            </h2>
                            <div style={{ marginTop: '5%' }}>
                                <div style={{ display: 'flex', justifyContent: 'center' }}>
                                {insight.top_titles_for_positive.map((title, index) => (
                                    <div
                                    key={index}
                                    style={{
                                    backgroundColor: '#39D069', /* Background color for the boxes */
                                    color: '#fff', /* Text color */
                                    padding: '5px 10px', /* Adjust padding as needed */
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
                            <div style={{  marginLeft: '10%', marginTop: '4%' }}>
                            <div style={{ width: '90%', backgroundColor: '#EBE5E5', borderRadius: '15px' }}>
                            <h2  style={{ fontSize: '30px', fontWeight: 'bold', marginLeft: '25%' }} >Recommended Actions</h2>
                            {insight.recommendations.map((recommendation, index) => (
                                <div style={{ fontSize: '25px', marginLeft: '4%' }} key={index}><CheckCircleOutlineIcon style={{ color: 'green'}} />{recommendation}</div>
                            ))}
                            </div>
                            <Button id="btns" variant="contained" sx={{ height: '50px', backgroundColor: '#E0E4FD', color: 'black', marginLeft: '18%', marginTop: '3%' }} onClick={saveResultsToExcel}>
                            Download Excel File
                            </Button>
                            <Button
                              id="btns"
                              variant="contained"
                              sx={{ height: '50px', backgroundColor: '#E0E4FD', color: 'black', marginTop: '3%', marginLeft: '2%' }}
                              onClick={handleSaveAsPdf}
                            >
                              Save Insight as PDF
                            </Button>
                        </div>
                            )
                        }     
                        </Box>
                    </div>
                    </div>
                    )}
                    </Box>
                </div>)
            )}
            <Box style={{ marginTop: result.length > 1 ? '0%' : '35%' }}>
              <Footer />
            </Box>
        </>
    )
}

export default Social
