import React from 'react'
import Navbar from '../components/Navbar'
import img1 from '../assets/img/AU1.png'
import img2 from '../assets/img/AU2.png'
import img3 from '../assets/img/AU3.png'
import img4 from '../assets/img/AU4.png'
import '../assets/css/Nav.css'


const About = () => {
    return (
        <>
        <Navbar />
        <br />
        <br /><br />
        <img id='abimg' src={img1} alt="img" srcset="" />
        <img id='abimg' src={img2} alt="img" srcset="" />
        <img id='abimg' src={img3} alt="img" srcset="" />
        <img id='abimg' src={img4} alt="img" srcset="" />
        </>
    )
}


export default About
