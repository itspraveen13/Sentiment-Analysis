import { Box } from '@mui/material'
import React, { useRef } from 'react';
import mainimg from '../assets/img/1.png'
import '../assets/css/main.css'
import { Link } from 'react-router-dom'
import Features from './Features'
import About from './About'

const Main = () => {
    const featuresRef = useRef(null);
    const scrollToFeatures = () => {
        featuresRef.current.scrollIntoView({ behavior: 'smooth' });
      };
    return (
        <>
            <Box sx={{ display: "flex", flexWrap: "wrap-reverse" }}>
                <img id='mainimg' style={{ width: "40%", margin: "5% 0 0 10%" }} src={mainimg} alt="img" />
                <Box sx={{ color: "#404A86" }}>
                    <h1 id="title" style={{ padding: "30% 10% 0 10%", fontSize: "35px" }}>Get Insights Using Sentiment Analysis</h1>
                    <h3 id="title2" style={{ padding: "5% 10% 0 10%" }}>Understand your customers like never before</h3>
                    <Box sx={{ display: { xs: "none", sm: "block" } }}>
                        <Link id="Link" onClick={scrollToFeatures}><h1 style={{ background: "#404A86", color: "white", padding: "10px 1px", margin: "5% 10%", borderRadius: "50px", textAlign: "center", fontSize: '20px', width: '150px',height: '50px' }} id='navitems1'>Explore &gt;</h1></Link>
                    </Box>
                </Box>
            </Box>
            <Box sx={{ display: { xs: "block", sm: "none" } }}>
                <Link id="Link" onClick={scrollToFeatures}><h1  style={{ background: "#404A86", color: "white", padding: "10px 0px", margin: "5% 30%", borderRadius: "50px", textAlign: "center", fontSize: '20px', width: '150px',height: '50px' }} id='navitems1'>Explore &gt;</h1></Link>
            </Box>
            <Box id="mainDiv" ref={featuresRef}>
                <Features />
            </Box>
            <Box  style={{ marginTop: '21%' }} >
                <About/>
            </Box>
        </>
    )
}

export default Main
