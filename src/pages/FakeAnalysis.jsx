import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import { Box, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, CircularProgress } from '@mui/material';
import axios from 'axios';
import { TypeAnimation } from 'react-type-animation';
import fake from '../assets/img/fake.png';

const FakeAnalysis = () => {
    const [file, setFile] = useState(null);
    const [text, setText] = useState('');
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false); // Added isLoading state

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleTextChange = (e) => {
        setText(e.target.value);
    };

    const handleTextSubmit = () => {
        setIsLoading(true); // Start loading animation
        axios
            .post('https://fake-review.up.railway.app/text', { text })
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

    const handleFileSubmit = () => {
        setIsLoading(true); // Start loading animation
        const formData = new FormData();
        formData.append('file', file);
        axios
            .post('https://fake-review.up.railway.app/data', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            })
            .then((res) => {
                console.log(res.data);
                setResults(res.data);
            })
            .catch((err) => {
                console.log(err);
            })
            .finally(() => {
                setIsLoading(false); // Stop loading animation
            });
    };

    return (
        <>
            <Navbar />
            <Box sx={{ margin: '5% 10%' }}>
                <Box sx={{ backgroundColor: 'white', borderRadius: '10px', padding: '10%', width: '80%' }}>
                    <TypeAnimation
                        sequence={[
                            'Type Your Text or Upload Dataset',
                            2000, // Waits 2s
                            () => ' ', // Clears the text
                            100, // Waits 2s
                            'Type Your Text or Upload Dataset', // Types the text again
                            () => {
                                console.log('Sequence completed');
                            },
                        ]}
                        wrapper="span"
                        cursor={true}
                        repeat={Infinity}
                        style={{ fontSize: '2em', display: 'inline-block', color: '#121e36' }}
                    />
                    <input
                        style={{ fontSize: '100%', paddingLeft: '20px', backgroundColor: '#E0E4FD' }}
                        id="sainput"
                        placeholder="Example: Jio Fiber Network is Fast"
                        value={text}
                        onChange={handleTextChange}
                    />
                    <Box sx={{ display: 'flex', justifyContent: 'center', margin: '20px' }}>
                        <Button
                            variant="contained"
                            onClick={handleTextSubmit}
                            sx={{ height: '50px', background: '#E0E4FD', color: '#121e36' }}
                        >
                            Analyze Text
                        </Button>
                        <h3 style={{ margin: '0 10px' }}>OR</h3>
                        {file ? (
                            <Button
                                variant="contained"
                                onClick={handleFileSubmit}
                                sx={{ height: '50px', backgroundColor: '#E0E4FD', color: '#121e36' }}
                            >
                                Analyze Dataset
                            </Button>
                        ) : (
                            <label htmlFor="file">
                                <Button
                                    component="span"
                                    variant="contained"
                                    sx={{ height: '50px', backgroundColor: '#E0E4FD', color: '#121e36' }}
                                >
                                    Upload File
                                </Button>
                                <input
                                    type="file"
                                    id="file"
                                    name="file"
                                    style={{ display: 'none' }}
                                    onChange={handleFileChange}
                                />
                            </label>
                        )}
                    </Box>
                    {isLoading && <CircularProgress style={{ margin: '20px auto' }} />} {/* Show loading animation if isLoading is true */}
                    <img
                        src={fake}
                        alt="GIF"
                        style={{
                            width: '600px',
                            height: '500px',
                            position: 'absolute',
                            top: '35%',
                            right: '10px',
                            transform: 'translateY(-50%)',
                        }}
                    />
                    {results.length > 0 && (
                        <TableContainer component={Paper} style={{ marginTop: '20px', backgroundColor: '#9aa4e7' }}>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Text</TableCell>
                                        <TableCell align="center">Prediction</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {results.map((result, index) => (
                                        <TableRow key={index} style={{ backgroundColor: '#E0E4FD' }}>
                                            <TableCell>{result.text}</TableCell>
                                            {/* <TableCell>{result.text}</TableCell> */}
                                            <TableCell align="center">{result.prediction}</TableCell>
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
                    )}
                </Box>
            </Box>
        </>
    );
};

export default FakeAnalysis;
