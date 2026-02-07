import React from 'react'
import { Box } from '@mui/material'
import InsertEmoticonIcon from '@mui/icons-material/InsertEmoticon';
import '../assets/css/feature.css'
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import ConnectWithoutContactIcon from '@mui/icons-material/ConnectWithoutContact';
import DocumentScannerIcon from '@mui/icons-material/DocumentScanner';
import memeimg from '../assets/img/meme.png';
import { useNavigate  } from 'react-router-dom';
import BGimg from '../assets/img/BG_IMG.png'

const Features = () => {
    const navigate = useNavigate();
    const handleBoxClick = (path) => {
      navigate(path);
    };

    return (
        <>
            <div id="mainDiv" style= {{ marginTop: '5%', marginBottom: '15%' }}>
            <h1 style={{ textAlign: "center", padding: "5% 0", color: "#404A86" }}>Make Use Of Our Features For Analysing</h1>
            <Box sx={{ display: { sm: "block", xs: "none" } }}>
                <Box sx={{ display: "flex", flexWrap: "wrap", margin: "10px 15%", gap:1 }}>
                    <Box id="card" onClick={() => handleBoxClick('/sentiment')}>
                        <h3 style={{ marginRight: "20px", fontSize: '25px' }}>Sentiment Analysis</h3>
                        <InsertEmoticonIcon style={{ fontSize: '35px'}}/>
                    </Box> 
                    <Box id="card" onClick={() => handleBoxClick('/churn')}>
                        <h3 style={{ marginRight: "20px", fontSize: '25px' }}>Churn Prediction</h3>
                        <AutoGraphIcon style={{ fontSize: '35px'}}/>
                    </Box>
                    <Box id="card" onClick={() => handleBoxClick('/social')}>
                        <h3 style={{ marginRight: "20px", fontSize: '25px' }}>Social Listening</h3>
                        <ConnectWithoutContactIcon style={{ fontSize: '35px'}}/>
                    </Box>
                </Box>
                <Box sx={{ display: "flex", flexWrap: "wrap", margin: "30px 25%",gap:1  }}>
                    <Box id="card1" onClick={() => handleBoxClick('/meme')}>
                        <h3 style={{ marginRight: "20px", fontSize: '25px' }}>Meme Analysis</h3>
                        <img src={memeimg} alt="" style={{ height: "35px", width: "30px" }} /> 
                    </Box>
                    <Box id="card1" onClick={() => handleBoxClick('/fake')}>
                        <h3 style={{ marginRight: "20px", fontSize: '25px' }}>Fake Detection</h3>
                        <DocumentScannerIcon style={{ fontSize: '35px'}}/>
                    </Box>
                </Box>
            </Box>
            <div id='tex' style={{ position: "absolute", top: "1503px" }}>
              <img style={{ width: "1700px" }} src={BGimg} alt="img" />
            </div>
            <Box sx={{ display: { sm: "none", xs: "block" } }}>
                <Box sx={{ display: "flex", margin: "10px 15%",flexDirection:"column" }}>
                    <Box id="card3" onClick={() => handleBoxClick('/sentiment')}>
                        <h3 style={{ marginRight: "20px" }}>Sentiment Analysis</h3>
                        <InsertEmoticonIcon />
                    </Box>
                    <Box id="card3" onClick={() => handleBoxClick('/churn')}>
                        <h3 style={{ marginRight: "20px" }}>Churn Prediction</h3>
                        <AutoGraphIcon />
                    </Box>
                    <Box id="card3" onClick={() => handleBoxClick('/social')}>
                        <h3 style={{ marginRight: "20px" }}>Social Listening</h3>
                        <ConnectWithoutContactIcon />
                    </Box>
                    <Box id="card3" onClick={() => handleBoxClick('/meme')}>
                        <h3 style={{ marginRight: "20px" }}>Meme Analysis</h3>
                        <img src={memeimg} alt="" style={{ height: "30px", width: "30px" }} />
                    </Box>
                    <Box id="card3" onClick={() => handleBoxClick('/fake')}>
                        <h3 style={{ marginRight: "20px" }}>Fake Detection</h3>
                        <DocumentScannerIcon />
                    </Box>
                </Box>
            </Box>
            </div>
        </>
    )
}

export default Features