import React, { useState } from "react";
import Navbar from "../components/Navbar";
import {
  Box,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
} from "@mui/material";
import axios from "axios";
import { TypeAnimation } from "react-type-animation";
import fake from "../assets/img/fake.png";

const FakeAnalysis = () => {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
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
      .post("https://fake-review.up.railway.app/text", { text })
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
    formData.append("file", file);
    axios
      .post("https://fake-review.up.railway.app/data", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
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
      <Box sx={{ margin: "5% 10%" }}>
        <Box
          sx={{
            backgroundColor: "white",
            borderRadius: "10px",
            padding: "10%",
            width: "80%",
          }}
        >
          <Box
            sx={{
              display: "flex",
              gap: 4,
              alignItems: "flex-start",
              justifyContent: "space-between",
              flexWrap: "wrap",
            }}
          >
            <Box
              sx={{
                flex: "1 1 560px",
                maxWidth: "84%",
                backgroundColor: "white",
                borderRadius: "12px",
                padding: { xs: "6%", md: "8%" },
              }}
            >
              <TypeAnimation
                sequence={[
                  "Type Your Text or Upload Dataset",
                  2000,
                  () => " ",
                  100,
                  "Type Your Text or Upload Dataset",
                ]}
                wrapper="span"
                cursor={true}
                repeat={Infinity}
                style={{
                  fontSize: "28px",
                  display: "inline-block",
                  color: "#121e36",
                  fontWeight: 600,
                }}
              />
              <input
                placeholder="Example: Jio Fiber Network is Fast"
                value={text}
                onChange={handleTextChange}
                style={{
                  display: "block",
                  width: "100%",
                  marginTop: "18px",
                  padding: "14px 16px",
                  fontSize: "16px",
                  backgroundColor: "#E9EEF9",
                  borderRadius: "8px",
                  border: "1px solid rgba(0,0,0,0.06)",
                }}
              />
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "flex-start",
                  gap: 2,
                  alignItems: "center",
                  marginTop: "18px",
                }}
              >
                <Button
                  variant="contained"
                  onClick={handleTextSubmit}
                  sx={{
                    height: "44px",
                    background: "#E0E4FD",
                    color: "#121e36",
                    textTransform: "none",
                    fontWeight: 600,
                  }}
                >
                  Analyze Text
                </Button>
                <Box
                  component="span"
                  sx={{ color: "#667085", fontWeight: 600 }}
                >
                  OR
                </Box>
                {file ? (
                  <Button
                    variant="contained"
                    onClick={handleFileSubmit}
                    sx={{
                      height: "44px",
                      backgroundColor: "#E0E4FD",
                      color: "#121e36",
                      textTransform: "none",
                      fontWeight: 600,
                    }}
                  >
                    Analyze Dataset
                  </Button>
                ) : (
                  <label htmlFor="file">
                    <Button
                      component="span"
                      variant="contained"
                      sx={{
                        height: "44px",
                        backgroundColor: "#E0E4FD",
                        color: "#121e36",
                        textTransform: "none",
                        fontWeight: 600,
                      }}
                    >
                      Upload File
                    </Button>
                    <input
                      type="file"
                      id="file"
                      name="file"
                      style={{ display: "none" }}
                      onChange={handleFileChange}
                    />
                  </label>
                )}
              </Box>
              {isLoading && <CircularProgress sx={{ marginTop: 2 }} />}
            </Box>
            <Box
              sx={{
                flex: "0 0 360px",
                maxWidth: "36%",
                display: { xs: "none", md: "block" },
                position: "sticky",
                top: "84px",
                alignSelf: "flex-start",
              }}
            >
              <img
                src={fake}
                alt="illustration"
                style={{
                  width: "100%",
                  maxHeight: "60vh",
                  height: "auto",
                  borderRadius: "12px",
                  objectFit: "contain",
                }}
              />
            </Box>
          </Box>
          {results.length > 0 && (
            <>
              <Box sx={{ mt: 2, mb: 1 }}>
                <h3
                  style={{
                    margin: 0,
                    fontSize: 18,
                    fontWeight: 700,
                    color: "black",
                  }}
                >
                  Analysis results
                </h3>
              </Box>
              <TableContainer
                component={Paper}
                sx={{
                  marginTop: "20px",
                  backgroundColor: "#020202",
                  width: "100%",
                }}
              >
                <Table sx={{ minWidth: 780 }}>
                  <TableHead>
                    <TableRow>
                      <TableCell>Text</TableCell>
                      <TableCell align="center">Prediction</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {results.map((result, index) => (
                      <TableRow
                        key={index}
                        style={{ backgroundColor: "#E0E4FD" }}
                      >
                        <TableCell>{result.text}</TableCell>
                        {/* <TableCell>{result.text}</TableCell> */}
                        <TableCell align="center">
                          {result.prediction}
                        </TableCell>
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
            </>
          )}
        </Box>
      </Box>
    </>
  );
};

export default FakeAnalysis;
