import React, { useState } from "react";
import Navbar from "../components/Navbar";
import {
  Box,
  Button,
  CircularProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import axios from "axios";
import Paper from "@mui/material/Paper";
import { TypeAnimation } from "react-type-animation";
import gifs from "../assets/img/sen.gif";

const SentimentAnalysis = () => {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState([]);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAudio, setIsAudio] = useState(false);
  const [language, setLanguage] = useState("Tamil");

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setIsAudio(e.target.files[0].type.startsWith("audio"));
  };

  const handleTextAnalyseClick = () => {
    if (!text) {
      setError("Please enter a review");
      return;
    }
    setError(null);
    setIsLoading(true);

    axios
      .post("http://localhost:5000/text", { text })
      .then((response) => {
        console.log(response.data);
        setResult(
          response.data.predictions.map((predictions) => ({
            text: predictions.text,
            sentiment: predictions.sentiment,
            reason: predictions.reasons,
            confidence: predictions.confidence, // Add confidence to the result
          })),
        );
        setIsLoading(false);
      })
      .catch((error) => {
        console.error(error);
        setError("Error occurred while analyzing the text");
        setIsLoading(false);
      });
  };

  const handleFileAnalyseClick = () => {
    if (!file) {
      setError("Please upload a file");
      return;
    }
    setError(null);
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("language", language);

    if (isAudio) {
      axios
        .post("http://localhost:5000/audio", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
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
            })),
          );
          setIsLoading(false);
        })
        .catch((error) => {
          console.error(error);
          setError("Error occurred while analyzing the audio file");
          setIsLoading(false);
        });
    } else {
      axios
        .post("http://localhost:5000/dataset", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
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
            })),
          );
          setIsLoading(false);
        })
        .catch((error) => {
          console.error(error);
          setError("Error occurred while analyzing the file");
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
      <Box sx={{ margin: "4% 6%" }}>
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
              id="sainput"
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
                onClick={handleTextAnalyseClick}
                sx={{
                  height: "44px",
                  backgroundColor: "#E0E4FD",
                  color: "black",
                  textTransform: "none",
                  fontWeight: 600,
                }}
              >
                Analyse Text
              </Button>
              <Box component="span" sx={{ color: "#667085", fontWeight: 600 }}>
                OR
              </Box>
              {file ? (
                <>
                  <Button
                    variant="contained"
                    sx={{
                      height: "44px",
                      backgroundColor: "#E0E4FD",
                      color: "black",
                      textTransform: "none",
                      fontWeight: 600,
                    }}
                    onClick={handleFileAnalyseClick}
                  >
                    Start Analyse
                  </Button>
                  <Box
                    sx={{ display: "inline-flex", alignItems: "center", ml: 2 }}
                  >
                    <Box
                      sx={{
                        backgroundColor: "#f1f5ff",
                        px: 2,
                        py: "6px",
                        borderRadius: "8px",
                      }}
                    >
                      {file.name}
                    </Box>
                    <Button
                      onClick={() => setFile(null)}
                      sx={{ ml: 1, minWidth: 36, height: 36 }}
                    >
                      ✕
                    </Button>
                  </Box>
                </>
              ) : (
                <label htmlFor="file">
                  <Button
                    component="span"
                    variant="contained"
                    sx={{
                      height: "44px",
                      backgroundColor: "#E0E4FD",
                      color: "black",
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
              <Stack direction={"column"}>
                <Box sx={{ padding: "6px 6px" }}>
                  <input
                    style={{ width: "18px", height: "18px" }}
                    type="checkbox"
                    id="audioCheckbox"
                    checked={isAudio}
                    onChange={(e) => setIsAudio(e.target.checked)}
                  />
                  <label
                    htmlFor="audioCheckbox"
                    style={{ color: "#0f1724", marginLeft: "8px" }}
                  >
                    Audio file
                  </label>
                </Box>
                {isAudio && (
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    style={{
                      marginTop: "8px",
                      padding: "6px",
                      borderRadius: "6px",
                    }}
                  >
                    <option value="Tamil">Tamil</option>
                    <option value="French">French</option>
                    <option value="Spanish">Spanish</option>
                    <option value="German">German</option>
                    <option value="Italian">Italian</option>
                  </select>
                )}
              </Stack>
            </Box>
            {error && (
              <p style={{ color: "red", marginTop: "12px" }}>{error}</p>
            )}
            {isLoading ? (
              <CircularProgress sx={{ marginTop: 2 }} />
            ) : result.length ? (
              <>
                <Box sx={{ mt: 2, mb: 1 }}>
                  <h3
                    style={{
                      margin: 0,
                      fontSize: 18,
                      fontWeight: 700,
                      color: "#363636",
                    }}
                  >
                    Analysis results
                  </h3>
                </Box>
                <TableContainer component={Paper} sx={{ width: "100%" }}>
                  <Table sx={{ minWidth: 780 }} aria-label="caption table">
                    <TableHead>
                      <TableRow style={{ backgroundColor: "#9aa4e7" }}>
                        <TableCell style={{ fontWeight: "bold" }}>
                          Review
                        </TableCell>
                        <TableCell
                          align="center"
                          style={{ fontWeight: "bold" }}
                        >
                          Sentiment
                        </TableCell>
                        <TableCell
                          align="center"
                          style={{ fontWeight: "bold" }}
                        >
                          Reason
                        </TableCell>
                        <TableCell
                          align="center"
                          style={{ fontWeight: "bold" }}
                        >
                          Confidence
                        </TableCell>
                        {/* Add Confidence column */}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {result.map((res, index) => (
                        <TableRow
                          key={index}
                          style={{ backgroundColor: "#E0E4FD" }}
                        >
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
              </>
            ) : null}
          </Box>

          <Box
            sx={{
              flex: "0 0 360px",
              maxWidth: "36%",
              display: { xs: "none", md: "block" },
            }}
          >
            <img
              className="fixed-illustration"
              src={gifs}
              alt="illustration"
              style={{
                width: 320,
                borderRadius: "12px",
                objectFit: "contain",
              }}
            />
          </Box>
        </Box>
      </Box>
    </>
  );
};

export default SentimentAnalysis;
