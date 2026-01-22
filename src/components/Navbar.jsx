import { AppBar, Box, Checkbox, Drawer, Stack, TextField } from "@mui/material";
import React, { useState } from "react";
import "../assets/css/Nav.css";
import logo from "../assets/img/icon.png";

// import '../assets/css/central.css'
//Icon
import MenuIcon from "@mui/icons-material/Menu";
import CloseIcon from "@mui/icons-material/Close";
import Modal from "@mui/material/Modal";

const style = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: 400,
  bgcolor: "background.paper",
  border: "2px solid #000",
  boxShadow: 24,
  p: 4,
};

const Navbar = () => {
  const [fun, setfun] = useState(false);
  const [open, setOpen] = React.useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);
  const [open1, setOpen1] = React.useState(false);
  const handleOpen1 = () => setOpen1(true);
  const handleClose1 = () => setOpen1(false);
  return (
    <>
      <AppBar sx={{ boxShadow: "none" }} id="navbar">
        <Box sx={{ display: { md: "block", sm: "none", xs: "none" } }}>
          <Stack
            direction="row"
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "0 40px",
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center" }}>
              <a
                href="/"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  textDecoration: "none",
                }}
              >
                <img
                  src={logo}
                  alt="Sentimetryx Logo"
                  style={{ height: "35px", cursor: "pointer" }}
                />
                <h2
                  style={{
                    fontFamily: "'Oswald', sans-serif !important",
                    fontSize: "28px",
                    fontWeight: "600",
                    letterSpacing: "1px",
                    lineHeight: "1",
                  }}
                >
                  <span style={{ color: "#ffffff" }}>Sentiment</span>
                  <span style={{ color: "#1495e5" }}>rix</span>
                </h2>
              </a>
            </Box>
            <Box sx={{ display: "flex", justifyContent: "space-between" }}>
              <a href="/">
                <h3 id="navitem2">Home</h3>
              </a>
              <a href="/about">
                <h3 id="navitem2">About us</h3>
              </a>
              <h3 id="navitem1" onClick={handleOpen}>
                Login
              </h3>
              <h3 id="navitem1" onClick={handleOpen1}>
                Signup
              </h3>
            </Box>
          </Stack>
        </Box>
        <Box sx={{ display: { md: "none", sm: "block", xs: "block" } }}>
          <Stack
            direction={"row"}
            sx={{ display: "flex", justifyContent: "space-between" }}
          >
            {/* <h3 id='navitem1' onClick={handleOpen}>Login</h3> */}
            <MenuIcon id="menuicon" onClick={(e) => setfun(true)} />
          </Stack>
        </Box>
      </AppBar>

      <Drawer anchor={"left"} open={fun} onClose={(e) => setfun(false)}>
        <CloseIcon id="closeicon" onClick={(e) => setfun(false)} />
        <Stack direction={"column"} sx={{ marginTop: "30%" }}>
          <a href="/">
            <h3 id="navitem2">Home</h3>
          </a>
          <a href="/about">
            <h3 id="navitem2">About us</h3>
          </a>
          <h3
            id="navitem2"
            onClick={() => {
              handleOpen();
              setfun(false);
            }}
          >
            Login
          </h3>
          <h3
            id="navitem2"
            onClick={() => {
              handleOpen1();
              setfun(false);
            }}
          >
            Signup
          </h3>
        </Stack>
      </Drawer>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box
          sx={{
            backgroundColor: "white",
            margin: { sm: "5% 30%", xs: "10% 10%" },
            padding: "0px 5% 5% 5%",
            borderRadius: "15px",
          }}
        >
          <CloseIcon
            sx={{
              color: "black",
              fontSize: "40px",
              margin: { sm: "10px 0px 0px 100%", xs: "10px 0px 0px 90%" },
            }}
            onClick={handleClose}
          />
          <h1 style={{ fontSize: "20px", textAlign: "center", color: "black" }}>
            Create an Account
          </h1>
          <div>
            <Box
              sx={{
                display: "flex",
                margin: "5% 0px",
                flexDirection: { sm: "row", xs: "column" },
              }}
            >
              <Box>
                <h3 id="formitem">First Name</h3>
                <input id="forminput" placeholder="Name" />
              </Box>
              <Box
                sx={{
                  margin: { sm: "0px 0px 0px 20%", xs: "10px 0px 0px 0px" },
                }}
              >
                <h3 id="formitem">Last Name</h3>
                <input id="forminput" placeholder="Last Name" />
              </Box>
            </Box>
            <Box
              sx={{
                display: "flex",
                margin: "5% 0px",
                flexDirection: { sm: "row", xs: "column" },
              }}
            >
              <Box>
                <h3 id="formitem">Email</h3>
                <input id="forminput" placeholder="Email" />
              </Box>
              <Box
                sx={{
                  margin: { sm: "0px 0px 0px 20%", xs: "10px 0px 0px 0px" },
                }}
              >
                <h3 id="formitem">Phone Number</h3>
                <input id="forminput" placeholder="Phone Number" />
              </Box>
            </Box>
            <Box
              sx={{
                display: "flex",
                margin: "5% 0px",
                flexDirection: { sm: "row", xs: "column" },
              }}
            >
              <Box>
                <h3 id="formitem">Password</h3>
                <input id="forminput" placeholder="Password" />
              </Box>
              <Box
                sx={{
                  margin: { sm: "0px 0px 0px 20%", xs: "10px 0px 0px 0px" },
                }}
              >
                <h3 id="formitem">Confirm Password</h3>
                <input id="forminput" placeholder="Confirm Password" />
              </Box>
            </Box>
            <Box sx={{ display: "flex", margin: "5% 0px" }}>
              <Checkbox defaultUnChecked />
              <h5 style={{ paddingTop: "10px" }}>
                {" "}
                Accept
                <span style={{ color: "#00426B" }}> Terms and conditions</span>
              </h5>
            </Box>
            <Box sx={{ display: "flex", justifyContent: "center" }}>
              <button
                style={{
                  color: "white",
                  backgroundColor: "#00426B",
                  borderRadius: "30px",
                  fontSize: "15px",
                  padding: " 2% 5%",
                }}
              >
                Create Account
              </button>
            </Box>
          </div>
        </Box>
      </Modal>
      <Modal
        open={open1}
        onClose={handleClose1}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box
          sx={{
            backgroundColor: "white",
            margin: { sm: "5% 30%", xs: "10% 10%" },
            padding: "0px 5% 5% 5%",
            borderRadius: "15px",
          }}
        >
          <CloseIcon
            sx={{
              color: "black",
              fontSize: "40px",
              margin: { sm: "10px 0px 0px 100%", xs: "10px 0px 0px 90%" },
            }}
            onClick={handleClose1}
          />
          <h1 style={{ fontSize: "20px", textAlign: "center", color: "black" }}>
            Login
          </h1>
          <div>
            <TextField
              style={{ width: "100%" }}
              id="outlined-basic"
              label="Email"
              variant="outlined"
            />
            <br />
            <br />
            <TextField
              style={{ width: "100%" }}
              id="outlined-basic"
              label="Password"
              variant="outlined"
            />
            <Box sx={{ display: "flex", justifyContent: "space-between" }}>
              <Box sx={{ display: "flex", margin: "5% 0px" }}>
                <Checkbox defaultUnChecked />
                <h5
                  style={{
                    paddingTop: "10px",
                    fontSize: "15px",
                    fontWeight: "400",
                  }}
                >
                  Remember Me
                </h5>
              </Box>
              <h5
                style={{
                  paddingTop: "8%",
                  fontSize: "15px",
                  fontWeight: "400",
                }}
              >
                Forgot Password ?
              </h5>
            </Box>
            <Box sx={{ display: "flex", justifyContent: "center" }}>
              <button
                style={{
                  color: "white",
                  backgroundColor: "#00426B",
                  borderRadius: "30px",
                  fontSize: "15px",
                  padding: " 2% 15%",
                }}
              >
                Login
              </button>
            </Box>
          </div>
        </Box>
      </Modal>
    </>
  );
};

export default Navbar;
