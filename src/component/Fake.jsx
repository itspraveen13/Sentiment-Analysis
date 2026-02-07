import React, { useState, useEffect, useRef } from 'react';
import { Box, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, CircularProgress } from '@mui/material';
import axios from 'axios';
import * as XLSX from 'xlsx';
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { storage } from '../assets/js/config.js';
import ClearIcon from '@mui/icons-material/Clear';
import Lottie from 'lottie-react';
import animationData from '../assets/js/Animation.json';
import BGimg from '../assets/img/BG_IMG.png'
import Footer from '../component/Footer.jsx'

const Fake = () => {
    const [file, setFile] = useState(null);
    const [text, setText] = useState('');
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false); // Added isLoading state
    const [excelData, setExcelData] = useState(null);
    const [inputText, setInputText] = useState('');
    const resultBoxRef = useRef(null);
    const resultBoxRef1 = useRef(null);

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
    
    useEffect(() => {
        if (excelData) {
          saveExcelToFirebase();
        }
    }, [excelData]);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        handleFileSubmit(e.target.files[0]);
    };

    const handleTextSubmit = () => {
        setIsLoading(true); // Start loading animation
        axios
            .post('http://localhost:5000/fake_text', { text })
            .then((res) => {
                console.log(res.data.prediction);
                setResults([{ text, prediction: res.data.prediction }]);
            })
            .catch((err) => {
                console.log(err);
            })
            .finally(() => {
                setIsLoading(false); // Stop loading animation
            });
    };

    const handleFileSubmit = (file) => {
        setIsLoading(true); // Start loading animation
        const formData = new FormData();
        formData.append('file', file);
        axios
            .post('http://localhost:5000/fake_file', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            })
            .then((res) => {
                console.log(res.data);
                setResults(res.data);
                setExcelData(res.data);
            })
            .catch((err) => {
                console.log(err);
            })
            .finally(() => {
                setIsLoading(false); // Stop loading animation
            });
    };

    const generateExcelFile = () => {
        const data = results.map((result) => ({
          Text: result.text,
          Prediction: result.prediction,
        }));
      
        const ws = XLSX.utils.json_to_sheet(data);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, 'FakeData');
        const currentDateIST = new Date().toLocaleString('en-IN', {
            timeZone: 'Asia/Kolkata',
          });
          
          // Splitting the date and time parts
          const parts = currentDateIST.split(', ');
          const datePart = parts[0].replace(/ /g, '-');
          const timePart = parts[1].replace(/:/g, '.');
                
          // Removing spaces between time and AM/PM and adding a hyphen
          const timeAMPM = timePart.replace(/\s+/g, '') + '-';
                
          const uniqueFileName = `${datePart}-${timeAMPM}fake_analysis_file.xlsx`;
                
          // Replacing slashes with periods
          const uniqueFileNameProcessed = uniqueFileName.replace(/\//g, '.');

          XLSX.writeFile(wb, uniqueFileNameProcessed);
      };

    const saveExcelToFirebase = async () => {
        const loggedUser = localStorage.getItem('username');
        if (loggedUser) {
        const data = results.map((result) => ({
            Text: result.text,
            Prediction: result.prediction,
          }));
        
          const ws = XLSX.utils.json_to_sheet(data);
          const wb = XLSX.utils.book_new();
          XLSX.utils.book_append_sheet(wb, ws, 'FakeData');
          const currentDateIST = new Date().toLocaleString('en-IN', {
              timeZone: 'Asia/Kolkata',
            });
            
            // Splitting the date and time parts
            const parts = currentDateIST.split(', ');
            const datePart = parts[0].replace(/ /g, '-');
            const timePart = parts[1].replace(/:/g, '.');
                  
            // Removing spaces between time and AM/PM and adding a hyphen
            const timeAMPM = timePart.replace(/\s+/g, '') + '-';
                  
            const uniqueFileName = `${datePart}-${timeAMPM}fake_analysis_file.xlsx`;
                  
            // Replacing slashes with periods
            const uniqueFileNameProcessed = uniqueFileName.replace(/\//g, '.');
                 
            const storageRef = ref(storage, `data/${loggedUser}/${uniqueFileNameProcessed}`);
            const blob = XLSX.write(wb, { type: "array", bookType: 'xlsx' });
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

    const handleClearResults = () => {
        window.scrollTo(0, 0);
        setResults([]);
        setInputText("");
    }
 
    return (
        <>
        <Box id="custombox">
            <h2 id='boxheader'>Fake Review Detection</h2>
            <h4 id='boxheader2'>Type Your Text, Upload Dataset</h4>
            <div style={{ display: "flex", justifyContent: "center" }}>
                <input
                    style={{ width: "400px" }}
                    id='boxsearch'
                    placeholder='example: Jio Fiber Network is Fast'
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                />
                {results != null && results.length >= 1 ? (
                    <ClearIcon style={{ marginTop: '4%' }} onClick={handleClearResults} />
                ) : (
                    null
                )}
            </div>
            <div style={{ display: "flex", justifyContent: "center" }}>
                <div style={{ display: "flex", flexDirection: "row", gap: 10 }}>
                    <Button id='btns' onClick={handleTextSubmit} disabled={!inputText.trim()}>
                        Analyse
                    </Button>
                        <label htmlFor="file" style={{ display: 'flex', alignItems: 'center' }}>
                            <Button id='btns' component="span" variant="contained" disabled={isLoading}>
                                Upload
                            </Button>
                            <input type="file" id="file" name="file" style={{ display: 'none' }} onChange={handleFileChange} accept=".xlsx, .csv, .mp3, .wav" />
                        </label>
                </div>
            </div>
            <h5 style={{ textAlign: "center", paddingTop: "30px", fontWeight: "400" }}>
                (For File Upload only xlsx, csv format files are accepted)
            </h5>
        </Box>
        <div id='tex' style={{ position: "absolute", top: "550px" }}>
            <img style={{ width: "1700px" }} src={BGimg} alt="img" />
        </div>
            {isLoading && 
            (
            <div style={{ height: '150px' }}>
                <Lottie animationData={animationData} loop={true} />
              </div>
            )}
                    {results != null && results.length > 1 ? (
                        <Box ref={resultBoxRef1} style={{ marginLeft: '21.5%', marginTop: '25%' }}>
                            <div>
                                <TableContainer component={Paper} sx={{ width: '866px' }} style={{ marginTop: '20px', backgroundColor: '#404A86' }}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell align="center" style={{fontSize: '20px', fontWeight: 'bold', paddingLeft: '43%', color: 'white'}}>Text</TableCell>
                                                <TableCell align="center" style={{fontSize: '20px', fontWeight: 'bold', paddingLeft: '55%', color: 'white'}}>Prediction</TableCell>
                                            </TableRow>
                                        </TableHead>
                                    </Table>
                                </TableContainer>
                            </div>
                            <div style={{ maxHeight: '600px', overflowY: 'auto', width: '866px'}}>
                                <TableContainer>
                                    <Table>
                                    <TableBody>
                                        {results.map((result, index) => (
                                            <TableRow key={index} style={{ backgroundColor: 'white' }}>
                                                <TableCell style={{ border: '2px solid #404A86' }}>{result.text}</TableCell>
                                                <TableCell  align="center" style={{paddingRight: '', border: '2px solid #404A86'}}>{result.prediction}</TableCell>
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
                            </div>
                            <Button
                                id='btns'
                                className="formbold-btn w-full"
                                style={{ marginLeft: '30%', marginTop: '3%', marginBottom: '5%' }}
                                onClick={generateExcelFile}
                            >
                                Download Excel
                            </Button> 
                        </Box>
                    ): (  <div>
                        {results.length === 1 && (
                        <Box ref={resultBoxRef} id="custombox1" style={{ borderColor: results[0].prediction === 'real' ? '#B9EACB' : 'red', borderWidth: '2px', borderStyle: 'solid' }}>
                            <div>
                                <h2 align="center">
                                    The Given Sentence is {results[0].prediction}
                                </h2>
                            </div>
                        </Box>
                      )}
                      </div>)
                    }
            <Box style={{ marginTop: results.length >= 1 ? '20%' : '30%' }}>
              <Footer />
            </Box>
        </>
    )
}

export default Fake
