import React from 'react'
import { Link } from 'react-router-dom'
import '../assets/css/about.css'
import FacebookIcon from '@mui/icons-material/Facebook';
import InstagramIcon from '@mui/icons-material/Instagram';
import TwitterIcon from '@mui/icons-material/Twitter';

const About = () => {
  return (
    <>
      <div style={{ background: "#404A86", padding: "50px 15% 0px 15%", bottom: 0 }}>
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
          {/* <div style={{color:"white"}}>
            <h2>TEAM</h2>
            <h5 style={{fontWeight:"400",color:"white"}}>Santhanabharathi </h5>
            <h5 style={{fontWeight:"400",color:"white"}}>Anni Sharleena </h5>
            <h5 style={{fontWeight:"400",color:"white"}}>Gaayathri </h5>
            <h5 style={{fontWeight:"400",color:"white"}}>Saranya </h5>
            <h5 style={{fontWeight:"400",color:"white"}}>Harini </h5>
          </div> */}
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