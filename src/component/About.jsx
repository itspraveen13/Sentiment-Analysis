import React from 'react'
import img4 from '../assets/img/4.png'
import icon1 from '../assets/img/icons/1.png'
import icon2 from '../assets/img/icons/2.png'
import icon3 from '../assets/img/icons/3.png'
import icon4 from '../assets/img/icons/4.png'
import icon5 from '../assets/img/icons/5.png'
import icon6 from '../assets/img/icons/6.png'
import icon7 from '../assets/img/icons/7.png'
import icon8 from '../assets/img/icons/8.png'
import icon9 from '../assets/img/icons/9.png'
import icon10 from '../assets/img/icons/10.png'
import icon11 from '../assets/img/icons/11.png'
import { Link } from 'react-router-dom'
import '../assets/css/about.css'
import FacebookIcon from '@mui/icons-material/Facebook';
import InstagramIcon from '@mui/icons-material/Instagram';
import TwitterIcon from '@mui/icons-material/Twitter';

const About = () => {
  return (
    <>
      <h1 id='s2' style={{ textAlign: "center" }}>What is Sentimetryx</h1>
      <div style={{ display: "flex", justifyContent: "center", paddingTop: "20px" }}>
        <img id='s4' style={{ width: "50%" }} src={img4} alt="main icon" />
      </div>
      <h4 id='s1' style={{ textAlign: "center", margin: "0px 150px", fontWeight: "400", paddingTop: "20px", fontSize: '20px'}}>Sentimetryx is your key to unlocking the pulse of customer sentiment. Leveraging cutting-edge technologies, it goes beyond traditional feedback analysis, identifying trends and patterns to help businesses stay ahead in the market. By decoding both positive and negative sentiments, Sentimetryx provides a holistic view, empowering companies to adapt and innovate. In a world where customer opinions shape market dynamics, Sentimetryx is the compass that guides businesses toward sustained growth and success.</h4>
      <h1 style={{ textAlign: "center", paddingTop: "50px" }}>Why Choose Us</h1>
      <div style={{ display: "flex", gap: 5, flexDirection: "row", flexWrap: "wrap", justifyContent: "center", marginTop: "30px" }}>
        <div style={{ background: "#404A86", padding: "30px", textAlign: "center", color: 'white', width: "250px" }}>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <img style={{ width: "50px", paddingBottom: "20px" }} src={icon1} alt="1" />
          </div>
          <h3>MULTILINGUAL SUPPORT</h3>
          <h5 style={{ fontWeight: "400" }}>Experience the power of effortless analysis across various languages with our multilingual support feature.</h5>
        </div>
        <div style={{ background: "#404A86", padding: "30px", textAlign: "center", color: 'white', width: "250px" }}>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <img style={{ width: "50px", paddingBottom: "20px" }} src={icon2} alt="1" />
          </div>
          <h3>VOICE ANALYSIS</h3>
          <h5 style={{ fontWeight: "400", marginTop: '15%' }}>Explore the emotional nuances of spoken feedback through our Voice Analysis feature.</h5>
        </div>
        <div style={{ background: "#404A86", padding: "30px", textAlign: "center", color: 'white', width: "250px" }}>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <img style={{ width: "50px", paddingBottom: "20px" }} src={icon3} alt="1" />
          </div>
          <h3>ACTIONABLE INSIGHTS</h3>
          <h5 style={{ fontWeight: "400" }}>Empower businesses with insightful analysis for strategic decisions and improved customer satisfaction.</h5>
        </div>
        <div style={{ background: "#404A86", padding: "30px", textAlign: "center", color: 'white', width: "250px" }}>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <img style={{ width: "50px", paddingBottom: "20px" }} src={icon4} alt="1" />
          </div>
          <h3>SUPPORT TEAM</h3>
          <h5 style={{ fontWeight: "400", marginTop: '16%' }}>A dedicated support team to ensure seamless user experiences and solve your queries.</h5>
        </div>

      </div>
      <div style={{ display: "flex", gap: 5, flexDirection: "row", flexWrap: "wrap", justifyContent: "center", marginTop: "10px" }}>
        <div style={{ background: "#404A86", padding: "30px", textAlign: "center", color: 'white', width: "250px" }}>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <img style={{ width: "50px", paddingBottom: "20px" }} src={icon5} alt="1" />
          </div>
          <h3>CHATBOT</h3>
          <h5 style={{ fontWeight: "400", marginTop: '2%' }}>Our Virtual Assistant Zen operates 24/7, answering questions and resolving user issues. Users can also reach out by filling out a form.</h5>
        </div>
        <div style={{ background: "#404A86", padding: "30px", textAlign: "center", color: 'white', width: "250px" }}>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <img style={{ width: "50px", paddingBottom: "20px" }} src={icon6} alt="1" />
          </div>
          <h3>INTERACTIVE VISUALIZATION</h3>
          <h5 style={{ fontWeight: "400" }}>Dive into interactive dashboards with charts and word clouds. Save and download your visual results effortlessly as a report.</h5>
        </div>
        <div style={{ background: "#404A86", padding: "30px", textAlign: "center", color: 'white', width: "250px" }}>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <img style={{ width: "50px", paddingBottom: "20px" }} src={icon7} alt="1" />
          </div>
          <h3>REAL-TIME DATA ACCESS</h3>
          <h5 style={{ fontWeight: "400" }}>Access real-time data with Social Listening, analyzing YouTube and Reddit comments on the fly with your chosen keywords.</h5>
        </div>
        <div style={{ background: "#404A86", padding: "30px", textAlign: "center", color: 'white', width: "250px" }}>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <img style={{ width: "50px", paddingBottom: "20px" }} src={icon8} alt="1" />
          </div>
          <h3>Diverse data formats</h3>
          <h5 style={{ fontWeight: "400" }}>Seamlessly handle and analyze various data formats—text, image, CSV, MP3, and beyond.</h5>
        </div>
      </div>
      <div style={{ display: "flex", gap: 5, flexDirection: "row", flexWrap: "wrap", justifyContent: "center", marginTop: "1px" }}>
        <div style={{ display: "flex", alignItems: "center", padding: "30px", textAlign: "center", color: 'black', width: "430px" }}>
          <img style={{ width: "80px", height: "80px", marginRight: "20px", marginTop: '5%' }} src={icon11} alt="1" />
          <div>
            <h3>MISSION</h3>
            <h5 style={{ fontWeight: "400" }}>Experience the power of effortless analysis across various languages with our multilingual support feature.</h5>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", padding: "30px", textAlign: "center", color: 'black', width: "430px" }}>
          <img style={{ width: "80px", height: "80px", marginRight: "20px" }} src={icon10} alt="1" />
          <div>
            <h3>PROMISE</h3>
            <h5 style={{ fontWeight: "400" }}>Experience the power of effortless analysis across various languages with our multilingual support feature.</h5>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", padding: "30px", textAlign: "center", color: 'black', width: "430px" }}>
          <img style={{ width: "80px", height: "80px", marginRight: "20px" }} src={icon9} alt="1" />
          <div>
            <h3>VIBE</h3>
            <h5 style={{ fontWeight: "400" }}>Experience the power of effortless analysis across various languages with our multilingual support feature.</h5>
          </div>
        </div>
      </div>
      
      <div style={{ background: "#404A86", padding: "50px 15% 0px 15%" }}>
        <div style={{ display: "flex",flexWrap:"wrap",flexDirection:"row" }}>
        <div id='s8' style={{ color: "white",width:"40%" }}>
            <h2 style={{ marginLeft: '15%', marginTop: '15%', fontSize: '35px' }}>Get in touch</h2>
            <h5 id='s7' style={{ fontSize: '15px', fontWeight: "400", width: "40%", marginLeft: '18%', marginTop: '6%', textAlign: 'center' }}>
              Want to find out how Sentimetryx can solve problems specific to your business? Let's talk.
            </h5>          
          </div>
          <div id='s6' style={{ background: "white", borderRadius: 20, padding: "50px" }}>
            <div style={{ display: 'flex', gap: 4 ,flexWrap:"wrap" }}>
              <div>
                <h4>Name</h4>
                <input type="text" />
              </div>
              <div>
                <h4>Email</h4>
                <input type="text" />
              </div>
              <div>
                <h4>Phone Number</h4>
                <input type="text" />
              </div>
            </div>
            <div style={{ marginTop: "30px" }}>
              <h4>Message</h4>
              <input type="text" style={{ width: "100%", height: "50px" }} />
            </div>
            <div style={{ display: 'flex',justifyContent:"center" }}>
              <Link id="Link"><h1 style={{ background: "#404A86", color: "white", width: "80px" }} align = 'center' className="navbtn1" id='navitems1'>Send</h1></Link>
            </div>
          </div>
        </div>

        <div style={{display:"flex",justifyContent:"space-between",marginTop:"50px",flexWrap:"wrap",gap:20}}>
          {/* <div style={{color:"white"}}> */}
            {/* <h2>TEAM TECHANZ</h2>
            <h5 style={{fontWeight:"400",color:"white"}}>Santhanabharathi </h5>
            <h5 style={{fontWeight:"400",color:"white"}}>Anni Sharleena </h5>
            <h5 style={{fontWeight:"400",color:"white"}}>Gaayathri </h5>
            <h5 style={{fontWeight:"400",color:"white"}}>Saranya </h5>
            <h5 style={{fontWeight:"400",color:"white"}}>Harini </h5> */}
          {/* </div> */}
          {/* <div style={{color:"white"}}>
            <h2>MENTOR</h2>
            <h5 style={{fontWeight:"400",color:"white"}}>Rajesh Krishna Kaarchan Thathan </h5>
          </div>
          <div style={{color:"white"}}>
            <h2>COMPANY</h2>
            <h5 style={{fontWeight:"400",color:"white"}}>VIRTUSA</h5>
          </div> */}
        </div>
        <div style={{background:"white",height:"2px",width:"100%",margin:"20px 5px"}}></div>
        <div style={{display:"flex", justifyContent:"space-between"}}>
          <div style={{display:"flex",gap:5,color:"white"}} >
            <FacebookIcon/>
            <TwitterIcon/>
            <InstagramIcon/>  
          </div>
          {/* <h5 id='s5' style={{color:"white",fontWeight:"400",marginBottom:"20px"}}>© 2023 All Rights Reserved By Virtusa - Team Techanz</h5> */}
        </div>
      </div>
    </>
  )
}

export default About