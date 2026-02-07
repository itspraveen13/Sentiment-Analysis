import { AppBar, Checkbox, Box, Drawer, Stack, TextField, Modal, Grid, IconButton, InputAdornment } from '@mui/material'
import { useState, useEffect } from 'react'
import { getStorage, ref, uploadString } from "firebase/storage";
import { storage, database } from '../assets/js/config.js';
import { sendPasswordResetEmail, getAuth, onAuthStateChanged, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, updateProfile } from 'firebase/auth';
import { useLocation } from 'react-router-dom';

//Image
import loginimg from '../assets/img/1.png'
import Signimg from '../assets/img/1.png'
import TCimg from '../assets/img/TandC.png'
//Icon
import React from 'react';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import Button from '@mui/material/Button';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import PersonIcon from '@mui/icons-material/Person';
//Style
import '../assets/css/Navbar.css'
import { Link } from 'react-router-dom';
import '../assets/css/login.css'
import logo from '../assets/img/icon.png'
function Navbar() {
    const [fun, setfun] = useState(false)
    const [anchorEl, setAnchorEl] = React.useState(null);
    const [anchorElLogin, setAnchorElLogin] = React.useState(null);
    const open = Boolean(anchorEl);
    const handleClick = (event) => {
        setAnchorEl(event.currentTarget);
    };
    const handleClose = () => {
        setAnchorEl(null);
    };
    const [open0, setOpen0] = React.useState(false);
    const [open1, setOpen1] = React.useState(false);
    const handleSignup = () => setOpen0(true); 
    const handleClose0 = () => {setOpen0(false), setPassword(''), setError(null);};
    const handleLogin = () => setOpen1(true);
    const handleClose1 = () => {setOpen1(false), setPassword(''), setError(null), setResetPassword(false);};
    const handleSwitchModal = (currentModal) => {
        if (currentModal === 'login') {
          handleClose1(); 
          handleSignup(); 
          setError(null);
        } else if (currentModal === 'signup') {
          handleClose0(); 
          handleLogin(); 
          setError(null);
        }
      };
    const [error, setError] = useState(null);
    const auth = getAuth();
    const [loggedInUser, setLoggedInUser] = useState(null);
    const location = useLocation();
    const isHomeRoute = location.pathname === '/';
    const [resetPassword, setResetPassword] = useState(false); 
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [isChecked, setIsChecked] = useState(false);
    const [openTerms, setOpenTerms] = useState(false);
  
    const handleOpenTerms = () => {
      setOpenTerms(true);
    };
  
    const handleCloseTerms = () => {
      setOpenTerms(false);
    };

    const handleCheckboxChange = (e) => {
        setIsChecked(e.target.checked);
      };
      
      const handlePasswordChange = (event) => {
        setPassword(event.target.value);
      };
    
      const togglePasswordVisibility = () => {
        setShowPassword(!showPassword);
      };

      const handleForgotPassword = () => {
        setResetPassword(true);
        setError('')
      };
    
      const handleResetPassword = async(e) => {
        e.preventDefault()
        const emalVal = document.getElementById('email').value;
        sendPasswordResetEmail(database,emalVal).then(data=>{
            alert("Check your gmail")
            setResetPassword(false);
        }).catch(err=>{
            setError("Please type your mail")
        })
      };

      const checkLoggedInUser = () => {
        const storedDisplayName = localStorage.getItem('username');
    
        if (storedDisplayName) {
          setLoggedInUser({ displayName: storedDisplayName });
        }
        
      };
    
      useEffect(() => {
        checkLoggedInUser();
      
        // Check for a logged-in user in localStorage
        const storedUser = localStorage.getItem('username');
        if (storedUser) {
          setLoggedInUser(storedUser);
        }
      }, []);
    
      const createAcc = async (e) => {
        e.preventDefault();
        setError(null);
            try {
            const userName = document.getElementById('username').value;
            const email = document.getElementById('email').value;
            const passWord = document.getElementById('password').value;
        
            if (!userName || !email || !passWord) {
              throw new Error('Please fill in all fields.');
            }
            if (!isChecked) {
              setError('Please accept T&C.');
            }
            const userData = await createUserWithEmailAndPassword(database, email, passWord);
        
            // Set the user's display name
            await updateProfile(userData.user, {
              displayName: userName,
            });
            const userStorageRef = ref(storage, `data/${userName}/${userName}.txt`);
          
            // Create the content to write to the file
            const fileContent = `Name: ${userName}\nEmail: ${email}`;
        
            // Upload the content to the file
            await uploadString(userStorageRef, fileContent, 'raw');
            
            setLoggedInUser(userData);
            loginUser();
            handleClose0();
          } catch (error) {
            console.error(error);
            if (error.code === 'auth/invalid-email') {
              setError('Invalid Email.');
            }
            else if (error.code === 'auth/weak-password') {
              setError('Weak Password!');
            }
            else if (error.code === 'auth/email-already-in-use') {
              setError('Email already exist try different account');
            } 
            else {
              setError(error.message);
            }
          }
      };
    
      const loginUser = async () => {
        try {
          setError(null);
          const email = document.getElementById('email').value;
          const password = document.getElementById('password').value;
          if (!email || !password) {
            throw new Error('Please fill in all fields.');
          }
    
          const userCredential = await signInWithEmailAndPassword(auth, email, password);
          const user = userCredential.user.displayName;
          handleClose1();
          setLoggedInUser(user);
          window.location.reload(); 
          localStorage.setItem('username', user);
        } catch (error) {
          console.error(error);
          if (error.code === 'auth/invalid-login-credentials') {
            setError('Wrong Credentials, please check!');
          }
          else if(error.code ==='auth/invalid-email'){
            setError('Invalid Email, please check!');
          }
          else {
            setError('Please fill in all fields.');
          }
        }
      };
    
      const handleLogout = async () => {
        try {
          await signOut(auth);
          setLoggedInUser(null);
          localStorage.removeItem('username');
          window.location.reload(); 
        } catch (error) {
          console.error(error);
        }
      };

      const handleMenuClick = (event) => {
        setAnchorElLogin(event.currentTarget);
      };

      const handleMenuClose = () => {
        setAnchorElLogin(null);
      };

    return (
        <>
            <AppBar id='appbar' style={{ position: "relative" }}>
                <Box sx={{ display: { xs: "none", sm: "block" } }}>
                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <Link to='/' className="nav-logo" aria-label="Sentimentrix home">
                            <img id='logo' src={logo} alt="Sentimentrix" />
                            <span className="nav-logo-text">
                                Sentiment<span className="nav-logo-accent">rix</span>
                            </span>
                        </Link>
                        <div style={{ display: "flex", marginRight: '5%' }}>
                            <Link id="Link" to='/'><h1 id='navitems1'>Home</h1></Link>
                            <Link id="Link" to='/about'><h1 id='navitems1'>About us</h1></Link>
                            <div>
                                <Button
                                    className='drop'
                                    id="basic-button"
                                    aria-controls={open ? 'basic-menu' : undefined}
                                    aria-haspopup="true"
                                    aria-expanded={open ? 'true' : undefined}
                                    onClick={handleClick}
                                >
                                    Features
                                </Button>
                                <Menu sx={{ marginTop: "20px" }}
                                    id="basic-menu"
                                    anchorEl={anchorEl}
                                    open={open}
                                    onClose={handleClose}
                                    MenuListProps={{
                                        'aria-labelledby': 'basic-button',
                                    }}
                                >
                                    <Link id="Link" to='/sentiment'><MenuItem onClick={handleClose}>Sentimental Analysis</MenuItem></Link>
                                    <Link id="Link" to='/churn'><MenuItem onClick={handleClose}>Churn Prediction</MenuItem></Link>
                                    <Link id="Link" to='/social'><MenuItem onClick={handleClose}>Social Listening</MenuItem></Link>
                                    <Link id="Link" to='/fake'><MenuItem onClick={handleClose}>Fake Detection</MenuItem></Link>
                                    <Link id="Link" to='/meme'><MenuItem onClick={handleClose}>Meme Analysis</MenuItem></Link>
                                </Menu>
                            </div>
                              {loggedInUser ? (
                                <>
                                  <IconButton id='navitems1' style={{ color: '#404A86', marginTop: '3%' }} onClick={handleMenuClick}>
                                    <PersonIcon style={{ border: 'solid 3px #404A86', borderRadius: '20px' }}/>
                                  </IconButton>
                                  <Menu
                                  anchorEl={anchorElLogin}
                                  open={Boolean(anchorElLogin)}
                                  onClose={handleMenuClose}
                                >
                                   <MenuItem key="username">
                                    {localStorage.getItem('username')}
                                   </MenuItem>
                                   <MenuItem>
                                    <a style={{ color: 'black', textDecoration: 'none' }} href="/history">History</a>
                                   </MenuItem>
                                  <MenuItem onClick={handleLogout}>Logout</MenuItem>
                                </Menu>
                                </>
                                ) : (
                                  <>                    
                                  <Link id="Link" onClick={handleLogin}><h1 style={{ width: "100px" }} className="navbtn1" id='navitems1' align='center'>Login</h1></Link>
                                  <Link id="Link" onClick={handleSignup}><h1 style={{ background: "#404A86", color: "white", width: "100px" }} align="center" className="navbtn1" id='navitems1'>Signup</h1></Link>
                                  </>
                              ) }
                            </div>
                    </Box>
                </Box>
                <Box sx={{ display: { sm: "none", xs: "block" } }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <Link to='/' className="nav-logo" aria-label="Sentimentrix home">
                            <img id='logo' src={logo} alt="Sentimentrix" />
                            <span className="nav-logo-text">
                                Sentiment<span className="nav-logo-accent">rix</span>
                            </span>
                        </Link>
                        <MenuIcon id="menuicon" onClick={e => (setfun(true))} />
                    </div>
                </Box>
            </AppBar>
            <Drawer
                anchor={'left'}
                open={fun}
                onClose={e => (setfun(false))} >
                <div style={{ display: "flex" }}>
                <CloseIcon id="closeicon" onClick={e => (setfun(false))} />
                </div>
                <br />
                <br />
                <div style={{ width: "300px" }}>
                    <Link id="Link" to="/"><h1 id='navitems1'>Home</h1></Link>
                    <Link id="Link" to="/about"><h1 id='navitems1'>About us</h1></Link>
                    <Link id="Link" to="/sentiment"><h1 id='navitems1'>Sentimental Analysis</h1></Link>
                    <Link id="Link" to="/churn"><h1 id='navitems1'>Churn Prediction</h1></Link>
                    <Link id="Link" to="/social"><h1 id='navitems1'>Social Listening</h1></Link>
                    <Link id="Link" to="/fake"><h1 id='navitems1'>Fake Detection</h1></Link>
                    <Link id="Link" to="/meme"><h1 id='navitems1'>Meme Analysis</h1></Link>
                    {loggedInUser ? (
                      <>
                        <a id="Link" style={{ color: 'black' }} href="/history"><h1 id='navitems1'>History</h1></a>
                        <Link id="Link" style={{ color: 'black' }} onClick={handleLogout}><h1 style={{ width: "50px", marginTop: "20px" }} className="navbtn1" id='navitems1'>Logout</h1></Link>
                      </>
                    ) : (
                      <>                    
                      <Link id="Link" onClick={handleLogin}><h1 style={{ width: "100px", marginTop: "20px" }} className="navbtn1" align="center" id='navitems1'>Login</h1></Link>
                      <Link id="Link" onClick={handleSignup}><h1 style={{ background: "#404A86", color: "white", width: "100px" }}  className="navbtn1" id='navitems1'>Signup</h1></Link>
                      </>
                   ) }
                </div>
            </Drawer>
            <Modal
                open={open1}
                onClose={handleClose1}
                aria-labelledby="modal-modal-title"
                aria-describedby="modal-modal-description"
            >
                <div style={{marginBottom:"50%"}}>
                    <Box id="logincard" >
                    <CloseIcon sx={{ cursor: "pointer", color: "black", fontSize: "40px", margin: { sm: "10px 0px 0px 100%", xs: "10px 0px 0px 90%" } }} onClick={handleClose1} />
                        <Grid container spacing={2}>
                            <Grid item sx={8}>
                            <Box sx={{display:{sm:"block",xs:"none"}}}>
                                <img id='cardimg' src={loginimg} alt="img" />
                            </Box>
                            </Grid>
                            <Box>
                            <Grid item sx={4}>
                            <h1 style={{textAlign:"center", paddingBottom:"20px", color:"black"}}>Login</h1>
                                <Stack direction={'column'}>
                                    <TextField id="email" label="Email" placeholder='Enter your Email' style={{ width: '300px', marginLeft: '5%' }}  multiline/>
                                    <br />
                                    {!resetPassword && (
                                        <TextField
                                        style={{ width: '300px', marginLeft: '5%' }}
                                        id="password"
                                        label="Password"
                                        variant="outlined"
                                        type={showPassword ? 'text' : 'password'}
                                        value={password}
                                        onChange={handlePasswordChange}
                                        InputProps={{
                                            endAdornment: (
                                            <InputAdornment position="end">
                                                <IconButton onClick={togglePasswordVisibility}>
                                                {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                                                </IconButton>
                                            </InputAdornment>
                                            ),
                                        }}
                                        />)}
                                    {!resetPassword && (
                                    <a><h5 id='sub' style={{cursor:'pointer', marginLeft:"60%",fontWeight:"300",color:"black",padding:"10px 0px"}} onClick={handleForgotPassword}>Forgot Password ?</h5></a>
                                    )}
                                    {resetPassword ? (
                                    <button id="btn" onClick={handleResetPassword}  style={{ cursor: 'pointer'}}>Reset Password</button> ) : (
                                    <button id="btn" onClick={loginUser} style={{ cursor: 'pointer'}}>Login</button> ) }
                                    <a id='sub' style={{textAlign:"center",padding:"50px 0 0 0"}}><span>Create new account ?</span><span style={{color:"#404A86", cursor: 'pointer'}} onClick={() => handleSwitchModal('login')}> Signup</span></a>
                                </Stack>
                            </Grid>
                            <Box><h4 align='center' style={{ color: 'red', fontWeight: 'normal' }}>{error}</h4></Box>
                            </Box>
                        </Grid>
                    </Box>
                </div>
            </Modal>
            <Modal
                open={open0}
                onClose={handleClose0}
                aria-labelledby="modal-modal-title"
                aria-describedby="modal-modal-description"
            >
                <div style={{marginBottom:"50%"}}>
                    <Box id="logincard" >
                    <CloseIcon sx={{ cursor: "pointer", color: "black", fontSize: "40px", margin: { sm: "10px 0px 0px 100%", xs: "10px 0px 0px 90%" } }} onClick={handleClose0} />
                        <Grid container spacing={2}>
                        <Grid item sx={8}>
                        <Box sx={{display:{sm:"block",xs:"none"}}}>
                            <img id='cardimg' src={Signimg} alt="" srcset="" />
                            </Box>
                        </Grid>
                        <Grid item sx={4}>
                            <h1 style={{textAlign:"center",paddingBottom:"20px",color:"black"}}>Signup</h1>
                            <Stack direction={'column'}>
                                <TextField id='username' label="Username" placeholder='Enter your username'  multiline style={{ width: '280px', marginLeft: '5%' }}/>
                                <br />
                                <TextField id='email' label="Email" placeholder='Enter your Email' style={{ width: '280px', marginLeft: '5%' }} multiline />
                                <br />
                                <TextField
                                    style={{ width: '280px', marginLeft: '5%' }}
                                    id="password"
                                    label="Password"
                                    variant="outlined"
                                    type={showPassword ? 'text' : 'password'}
                                    value={password}
                                    onChange={handlePasswordChange}
                                    InputProps={{
                                        endAdornment: (
                                        <InputAdornment position="end">
                                            <IconButton onClick={togglePasswordVisibility}>
                                            {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                                            </IconButton>
                                        </InputAdornment>
                                        ),
                                    }}
                                    /> 
                                <br />
                                <Box sx={{ display: "flex", marginLeft: '10%' }}>
                                <Checkbox checked={isChecked} onChange={handleCheckboxChange} />
                                    <h5 style={{ marginTop: '4%'}}>  Accept<Link style={{ color: "#00426B" }} onClick={handleOpenTerms}> Terms and conditions</Link></h5>
                                </Box>
                                <button id="btn" style={{ cursor: 'pointer'}} onClick ={createAcc}>Signup</button>
                                <a id='sub' style={{textAlign:"center",padding:"30px 0 0 0"}}><span>Already have account ?</span><span onClick={() => handleSwitchModal('signup')} style={{color:"#404A86", cursor: 'pointer'}}> Login</span></a>
                            </Stack>
                            <Box><h4 align='center' style={{ color: 'red', fontWeight: 'normal' }}>{error}</h4></Box>
                        </Grid>
                        </Grid>
                    </Box>
                </div>
            </Modal>
            <Modal
              open={openTerms}
              onClose={handleCloseTerms}
              aria-labelledby="modal-modal-title"
              aria-describedby="modal-modal-description"
            >
              <Box id="tcCard" style={{ height: '550px', overflowY: 'auto', overflowX: 'hidden' }}>
                <CloseIcon
                  sx={{ cursor: 'pointer', color: 'black', fontSize: '40px', margin: { sm: '10px 0px 0px 100%', xs: '10px 0px 0px 90%' } }}
                  onClick={handleCloseTerms}
                />
                <img style={{ width: '99%' }} src={TCimg} alt="Terms and Conditions" />
              </Box>
            </Modal>
        </>
    )
}

export default Navbar
