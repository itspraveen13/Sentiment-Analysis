import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Box, Button } from '@mui/material';
import Grid from '@mui/material/Unstable_Grid2'; // Grid version 2
import { Chart } from 'react-google-charts';
import GroupsIcon from '@mui/icons-material/Groups';
import ThumbDownAltIcon from '@mui/icons-material/ThumbDownAlt';
import ThumbUpAltIcon from '@mui/icons-material/ThumbUpAlt';
import CurrencyRupeeIcon from '@mui/icons-material/CurrencyRupee';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import '../assets/css/common.css'
import Lottie from 'lottie-react';
import animationData from '../assets/js/Animation.json';
import BGimg from '../assets/img/BG_IMG.png'
import Footer from '../component/Footer.jsx'
import html2canvas from 'html2canvas';
import { saveAs } from 'file-saver';
import jsPDF from 'jspdf';

const Churn = () => {
    const [fileNames, setFileNames] = useState([]);
    const [result, setResult] = useState([]);
    const [responseMessage, setResponseMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showTable, setShowTable] = useState(false);
    const [error, setError] = useState(null);
    const [excelFileUrl, setExcelFileUrl] = useState(null);
    const [excelData, setExcelData] = useState(null);
    const [analysisCompleted, setAnalysisCompleted] = useState(null);
    const [pieChartData, setPieChartData] = useState([]);
    const [pieChartDataRevenue, setPieChartDataRevenue] = useState([]);
    const [barChartData, setBarChartData] = useState([]);
    const [barChartDataTop9, setBarChartDataTop9] = useState([]);
    const [uploadedFile, setUploadedFile] = useState(null);
    const [fileName, setFileName] = useState('');
    const resultBoxRef = useRef(null);
    const fileInputRef = useRef(null);

    const handleScrollIntoView = () => {
      if (resultBoxRef.current) {
        resultBoxRef.current.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }
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
          const pdf = new jsPDF('p', 'pt', [790,790]);
          pdf.addImage(canvas, 'PNG', 0, 0, scaledWidth, scaledHeight);
          const pdfBlob = pdf.output('blob');
          saveAs(pdfBlob, 'Churn.pdf');
        });
    };

    useEffect(() => {
      // Scroll into view when the component mounts and conditions are met
      if (result) {
        handleScrollIntoView();
      }
    }, [result, analysisCompleted]);

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

    const handleFileUpload = async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        setIsLoading(true);
        setAnalysisCompleted(false);
        try {
          const response = await axios.post('http://localhost:5000/churn', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });

          setResult(response.data);
          console.log(response.data);
          setExcelData(response.data);
          setResponseMessage('');
          setIsLoading(false);
          setShowTable(true);
          setAnalysisCompleted(true);
          console.log("uploaded :", uploadedFile);

          const barData = [
            ['Title', 'Churn', 'Retention'],
            ['Month-to-month', response.data.contract_counts_neg['Month-to-month'], response.data.contract_counts_pos['Month-to-month']],
            ['One year', response.data.contract_counts_neg['One year'], response.data.contract_counts_pos['One year']],
            ['Two year', response.data.contract_counts_neg['Two year'], response.data.contract_counts_pos['Two year']]
          ];

          const pieChart = [
            ['Category', 'Count'],
            ['Negative', response.data.num_negative],
            ['Positive', response.data.num_positive],
          ];

          const pieChartRevenue = [
            ['Category', 'Count'],
            ['Churn', response.data.charges_churn],
            ['Retention', response.data.charges_not_churn],
          ];

          const barDataTop9 = [['Reason', 'Frequency']];
          response.data.word_frequency.forEach(item => {
            barDataTop9.push(item);
          });

          setBarChartDataTop9(barDataTop9)
          setPieChartDataRevenue(pieChartRevenue)
          setBarChartData(barData)
          setPieChartData(pieChart)
        } catch (error) {
          console.error(error);
          const serverError = error?.response?.data?.error;
          setError(serverError ? `Error: ${serverError}` : 'Error occurred while analyzing');
          setIsLoading(false);
        }
      };
    
    const handleClearResults = () => {
      window.scrollTo(0, 0);
      setAnalysisCompleted(false);
      setResult([]);
      setUploadedFile(null);
  }

    const handleFileSelect = (event) => {
      const file = event.target.files?.[0];
      if (file) {
        handleFileUpload(file);
        setUploadedFile(file);
        setFileName(file.name);
      }
      event.target.value = '';
    };
    return (
        <>
          <Box style={{ height: '100' }}>    
              {uploadedFile ? (
                <>
                <Box id="custombox3" style={{ position: 'relative', zIndex: 2 }}>
                <h2 id='boxheader'>Uploaded File</h2>
                  <Box style={{ textAlign: 'center', marginTop: '5%' }}>
                    <p>File Name: {fileName}</p>
                    <Button
                      variant="outlined"
                      color="secondary"
                      onClick={handleClearResults}
                    >
                      Remove File
                    </Button>
                  </Box>
                </Box>
                <div id='tex' style={{ position: "absolute", top: "320px", pointerEvents: 'none', zIndex: 0 }}>
                    <img style={{ width: "1700px" }} src={BGimg} alt="img" />
                </div>
                </>
              ) : (
                <>
                <Box id="custombox2">
                  <h2 id='boxheader'>Churn Prediction</h2>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".xlsx,.csv"
                    style={{ display: 'none' }}
                    onChange={handleFileSelect}
                  />
                  <div id="dropbox" style={{ textAlign: 'center', padding: '20px' }}>
                    <Button
                      variant="outlined"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Upload File
                    </Button>
                  </div>
                  <h5 style={{ textAlign: "center", paddingTop: "30px", fontWeight: "400" }}>
                    (For File Upload only xlsx, csv format files are accepted and there should be a column as Sentiment)
                  </h5>
                  </Box>
                  <div id='tex' style={{ position: "absolute", top: "550px" }}>
                    <img style={{ width: "1700px" }} src={BGimg} alt="img" />
                  </div>
                </>
              )}
          </Box>
          {
            isLoading && (
              <div style={{ height: '150px', marginTop: '5%' }}>
              <Lottie animationData={animationData} loop={true} />
            </div>
            )
          } 
          <h4 align='center' style={{ color:'red', marginTop: '5%' }}>{error}</h4>
            { 
                analysisCompleted &&  (
                <Box  ref={resultBoxRef}>
                <div id="insightPDF">
                  <Box style={{ margin: '1%', marginTop: '21%' }}>
                    <h2 align="center">The Churn Rate of Your Company is {result.churn_rate.toFixed(2)}%</h2>
                  </Box>       
                  <div style={{ padding: '1%', backgroundColor:'#ddd'}}>
                  <Grid container spacing={2} style={{ margin: 0 }} id="mainGrid">
                    <Grid id="iconGrid" item xs={12} sm={6} md={4} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF', width: '550px' }}>
                    <Grid container spacing={2} style={{ margin: '1% 6%' }} id="mainGrid">
                      {/* First Column */}
                      <Grid item xs={12} sm={6} md={4} style={{ backgroundColor: '#FFFFFF', width: '230px' }}>
                        <div style={{ padding: '3%', textAlign: 'center' }}>
                          <GroupsIcon  style={{ fontSize:'50px' }}/>
                          <p>Total Subscribers</p>
                          {result.num_negative + result.num_positive}
                        </div>
                        <div style={{ padding: '3%', textAlign: 'center' }}>
                          <ThumbDownAltIcon style={{ color: 'red', fontSize:'50px' }} />
                          <p>Subscribers Churning</p>
                          {result.num_negative}
                        </div>
                        <div style={{ padding: '3%', textAlign: 'center' }}>
                          <ThumbUpAltIcon style={{ color: 'green', fontSize:'50px' }} />
                          <p>Subscribers Retained</p>
                          {result.num_positive}
                        </div>
                      </Grid>
                      {/* Second Column */}
                      <Grid item xs={12} sm={6} md={8} style={{  backgroundColor: '#FFFFFF', width: '200px' }}>
                        <div style={{ padding: '3%', textAlign: 'center' }}>
                          <CurrencyRupeeIcon style={{ fontSize:'50px' }} />
                          <p>Total Revenue</p>
                          {(result.charges_churn + result.charges_not_churn).toFixed(2)}
                        </div>
                        <div style={{ padding: '3%', textAlign: 'center' }}>
                          <TrendingDownIcon style={{ color: 'red', fontSize:'50px' }} />
                          <p>Revenue Lost</p>
                          {result.charges_churn.toFixed(2)}
                        </div>
                        <div style={{ padding: '3%', textAlign: 'center' }}>
                          <TrendingUpIcon style={{ color: 'green', fontSize:'50px' }} />
                          <p>Revenue Retained</p>
                          {result.charges_not_churn.toFixed(2)}
                        </div>
                      </Grid>
                  </Grid>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF', height: '400px', width: '450px' }}>
                      <h4 align= "center">Churn to Retention Proportion</h4>
                      <div style={{ padding: '10% 5%' }}>
                      <Chart
                            chartType="PieChart"
                            data={pieChartData}
                            options={{
                              chartArea: { width: '100%', height: '100%' },
                              colors: ['#FA2D2D', '#1DBB57'],
                              pieHole: 0.4,
                              backgroundColor: 'none',
                              dataLabels: {
                                color: 'red', // Set the color of the data labels
                              },
                              legend: { position: 'bottom' },
                            }}
                            width={'100%'} 
                            height={'250px'}
                          />
                      </div>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF', height: '400px', width: '485px' }}>
                    <h4 align= "center">Customer Churn vs Retention</h4>
                    <div style={{ padding: '2%' }}>
                    <Chart
                        chartType="BarChart"
                        width="100%"
                        height="300px"
                        data={barChartData}
                        options={{
                          chartArea: { width: '80%', height: '75%' },
                          hAxis: { title: 'Count' },
                          vAxis: { title: 'Title', minValue: 0 },
                          legend: 'none',
                          backgroundColor: 'none',
                          colors: ['#FA2D2D', '#1DBB57'],
                        }}
                      />
                    </div>
                    </Grid>
                  </Grid>
                  <Grid container spacing={2} style={{ margin: 0 }} id="mainGrid">
                    <Grid item xs={12} sm={4} md={4} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF', height: '400px', width: '430px' }}>
                    <h4 align= "center">Revenue Distribution</h4>
                      <div style={{ padding: '10% 5%' }}>
                      <Chart
                            chartType="PieChart"
                            data={pieChartDataRevenue}
                            options={{
                              chartArea: { width: '100%', height: '100%' },
                              colors: ['#FA2D2D', '#1DBB57'],
                              pieHole: 1,
                              backgroundColor: 'none',
                              dataLabels: {
                                color: 'red',
                              },
                              legend: { position: 'bottom' },
                              pieStartAngle: 120,
                            }}
                            width={'100%'} 
                            height={'250px'}
                          />
                      </div>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF', height: '400px', width: '600px' }}>
                      <h4 align='center'>Top 8 Reasons for Churn</h4>
                      <div style={{ marginLeft: '4%' }}>
                      <Chart
                        chartType="BarChart"
                        width="100%"
                        height="350px"
                        data={barChartDataTop9}
                        options={{
                          chartArea: { width: '70%', height: '75%' },
                          hAxis: { title: 'Frequency' },
                          vAxis: { title: 'Reason' },
                          legend: 'none',
                          backgroundColor: 'none',
                        }}
                      />
                      </div>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4} style={{ border: '5px solid #ddd', backgroundColor: '#FFFFFF', height: '400px', width: '453px' }}>
                      <h4 align='center'>Recommendations</h4>
                      {result.recommendation.map((recommendation, index) => (
                      <div key={index} style={{ display: 'flex', alignItems: 'center', marginTop: '3%' }}>
                      <h3 style={{ marginRight: '5px' }}>{index + 1}.</h3>
                      <h5 style={{ padding: '3%', fontWeight: 'normal' }}>{recommendation}</h5>
                      </div>
                    ))}
                    </Grid>
                  </Grid>
                  </div>
                </div>
                      <div style={{ display: 'flex', justifyContent: 'center', margin: '1%' }}>
                      <Button
                      id="btns"
                        variant="contained"
                        sx={{ height: '50px', backgroundColor: '#E0E4FD', color: 'black' }}
                        onClick={handleSaveAsPdf}
                      >
                        Save as PDF
                      </Button>
                    </div>
                </Box>
                )
            }
            <Box style={{ marginTop: analysisCompleted ? '0%' : '25%' }}>
              <Footer />
            </Box>
        </>
    )
}

export default Churn
