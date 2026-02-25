import { useState, useRef, useEffect, useCallback } from "react";

const GRADIENT_BG = "linear-gradient(155deg, #0a1628 0%, #162052 20%, #3b1760 45%, #7b1a5e 70%, #c2185b 95%)";

const BACKGROUNDS = [
  { id: "none", label: "None", color: null, preview: "#1a1a2e" },
  { id: "blur", label: "Blur", color: null, preview: "linear-gradient(135deg, #667eea, #764ba2)" },
  { id: "gradient1", label: "Sunset", color: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", preview: "linear-gradient(135deg, #f093fb, #f5576c)" },
  { id: "gradient2", label: "Ocean", color: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)", preview: "linear-gradient(135deg, #4facfe, #00f2fe)" },
  { id: "gradient3", label: "Forest", color: "linear-gradient(135deg, #0ba360 0%, #3cba92 100%)", preview: "linear-gradient(135deg, #0ba360, #3cba92)" },
  { id: "warm", label: "Warm", color: "linear-gradient(135deg, #f6d365 0%, #fda085 100%)", preview: "linear-gradient(135deg, #f6d365, #fda085)" },
];

const MAX_DURATION = 30;

// Canvas background compositing
function useBackgroundEffect(videoRef, canvasRef, selectedBg) {
  const animFrameRef = useRef(null);
  const ctxRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    ctxRef.current = canvas.getContext("2d", { willReadFrequently: true });

    const draw = () => {
      const ctx = ctxRef.current;
      if (!ctx || video.paused || video.ended) {
        animFrameRef.current = requestAnimationFrame(draw);
        return;
      }

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      if (selectedBg === "none") {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      } else if (selectedBg === "blur") {
        ctx.filter = "blur(12px)";
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.filter = "none";
        ctx.save();
        ctx.beginPath();
        ctx.ellipse(canvas.width / 2, canvas.height / 2, canvas.width * 0.32, canvas.height * 0.45, 0, 0, Math.PI * 2);
        ctx.clip();
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
      } else {
        const bg = BACKGROUNDS.find((b) => b.id === selectedBg);
        if (bg?.color) {
          if (bg.color.startsWith("linear")) {
            const grd = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
            grd.addColorStop(0, bg.color.match(/#[a-f0-9]{6}/gi)?.[0] || "#333");
            grd.addColorStop(1, bg.color.match(/#[a-f0-9]{6}/gi)?.[1] || "#666");
            ctx.fillStyle = grd;
          } else {
            ctx.fillStyle = bg.color;
          }
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.save();
          ctx.beginPath();
          ctx.ellipse(canvas.width / 2, canvas.height / 2, canvas.width * 0.32, canvas.height * 0.45, 0, 0, Math.PI * 2);
          ctx.clip();
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          ctx.restore();
        } else {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }
      }

      animFrameRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [selectedBg, videoRef, canvasRef]);
}

// ─── Shared Components ──────────────────────────────────────

function Logo() {
  return (
    <div style={styles.logo} className="logo">
      <span style={styles.logoText}>VideoVoice</span>
    </div>
  );
}

// ─── Screens ────────────────────────────────────────────────

function WelcomeScreen({ onStart }) {
  return (
    <div style={styles.gradientScreen} className="gradient-screen">
      <div style={styles.gradientOverlay} />
      <div style={styles.topSection}>
        <Logo />
      </div>
      <div style={styles.centerSection}>
        <h1 style={styles.welcomeHeading} className="welcome-heading">
          WHY DO YOU LOVE WORKING HERE?
        </h1>
      </div>
      <div style={styles.bottomAction}>
        <button onClick={onStart} style={styles.outlineBtn} className="outline-btn">
          RECORD YOUR ANSWER
        </button>
      </div>
    </div>
  );
}

function RecordScreen({ onNext, onBack }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const [phase, setPhase] = useState("setup");
  const [countdown, setCountdown] = useState(3);
  const [elapsed, setElapsed] = useState(0);
  const [selectedBg, setSelectedBg] = useState("none");
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState(null);

  useBackgroundEffect(videoRef, canvasRef, selectedBg);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: "user" },
          audio: true,
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
        setCameraReady(true);
      } catch (err) {
        setCameraError("Camera access denied. Please allow camera & microphone permissions.");
      }
    })();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const startCountdown = useCallback(() => {
    setPhase("countdown");
    let c = 3;
    setCountdown(c);
    const iv = setInterval(() => {
      c--;
      if (c <= 0) {
        clearInterval(iv);
        startRecording();
      } else {
        setCountdown(c);
      }
    }, 1000);
  }, []);

  const startRecording = useCallback(() => {
    chunksRef.current = [];
    const canvasStream = canvasRef.current.captureStream(30);
    const audioTrack = streamRef.current?.getAudioTracks()[0];
    if (audioTrack) canvasStream.addTrack(audioTrack);

    const mr = new MediaRecorder(canvasStream, { mimeType: "video/webm;codecs=vp9,opus" });
    mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      setRecordedBlob(blob);
      setRecordedUrl(URL.createObjectURL(blob));
      setPhase("preview");
    };
    mediaRecorderRef.current = mr;
    mr.start(100);
    setPhase("recording");
    setElapsed(0);

    timerRef.current = setInterval(() => {
      setElapsed((prev) => {
        if (prev + 1 >= MAX_DURATION) {
          stopRecording();
          return MAX_DURATION;
        }
        return prev + 1;
      });
    }, 1000);
  }, []);

  const stopRecording = useCallback(() => {
    clearInterval(timerRef.current);
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const retake = () => {
    setRecordedBlob(null);
    setRecordedUrl(null);
    setPhase("setup");
    setElapsed(0);
    if (videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play();
    }
  };

  const progress = (elapsed / MAX_DURATION) * 100;

  return (
    <div style={styles.cameraScreen} className="camera-screen">
      {/* Back button */}
      {(phase === "setup") && (
        <button onClick={onBack} style={styles.backBtn} className="back-btn">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
      )}
      {/* Camera view */}
      <div style={styles.cameraView} className="camera-view">
        <video ref={videoRef} style={styles.hiddenVideo} muted playsInline />
        <canvas ref={canvasRef} style={{ ...styles.cameraFeed, display: phase === "preview" ? "none" : "block" }} />
        {phase === "preview" && (
          <video src={recordedUrl} style={styles.cameraFeed} controls autoPlay loop />
        )}

        {cameraError && (
          <div style={styles.cameraErrorOverlay}>{cameraError}</div>
        )}

        {phase === "countdown" && (
          <div style={styles.countdownOverlay}>
            <span style={styles.countdownNum} className="countdown-num">{countdown}</span>
          </div>
        )}

        {phase === "recording" && (
          <>
            <div style={styles.recIndicator}>
              <span style={styles.recDot} />
              <span style={styles.recText}>{elapsed}s / {MAX_DURATION}s</span>
            </div>
            <div style={styles.progressBar}>
              <div style={{ ...styles.progressFill, width: `${progress}%` }} />
            </div>
          </>
        )}
      </div>

      {/* Bottom panel */}
      <div style={styles.bottomPanel} className="bottom-panel">
        {/* Background picker in setup */}
        {phase === "setup" && (
          <div style={styles.bgSection} className="bg-section">
            <p style={styles.bgTitle}>Choose your background</p>
            <div style={styles.bgThumbs} className="bg-thumbs">
              {BACKGROUNDS.map((bg) => (
                <button
                  key={bg.id}
                  onClick={() => setSelectedBg(bg.id)}
                  style={{
                    ...styles.bgThumb,
                    background: bg.preview,
                    ...(selectedBg === bg.id ? styles.bgThumbActive : {}),
                  }}
                  className="bg-thumb"
                  title={bg.label}
                />
              ))}
            </div>
          </div>
        )}

        {/* Controls */}
        <div style={styles.controlsRow} className="controls-row">
          {phase === "setup" && (
            <button
              onClick={startCountdown}
              disabled={!cameraReady}
              style={{ ...styles.recordBtn, opacity: cameraReady ? 1 : 0.4 }}
              className="record-btn"
            >
              <span style={styles.recordDot} />
            </button>
          )}
          {phase === "countdown" && (
            <button style={{ ...styles.recordBtn, opacity: 0.4 }} disabled className="record-btn">
              <span style={styles.recordDot} />
            </button>
          )}
          {phase === "recording" && (
            <button onClick={stopRecording} style={styles.recordBtn} className="record-btn">
              <span style={styles.stopSquare} />
            </button>
          )}
          {phase === "preview" && (
            <div style={styles.previewBtns} className="preview-btns">
              <button onClick={retake} style={styles.outlineBtn} className="outline-btn">RETAKE</button>
              <button onClick={() => onNext(recordedBlob)} style={styles.filledBtn} className="filled-btn">NEXT &rarr;</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function EmailScreen({ onNext, onBack, error: serverError }) {
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = () => {
    if (!email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      setError("Please enter a valid email");
      return;
    }
    setError("");
    onNext(email);
  };

  return (
    <div style={styles.gradientScreen} className="gradient-screen">
      <div style={styles.gradientOverlay} />
      <button onClick={onBack} style={styles.backBtnGradient} className="back-btn">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>
      <div style={styles.topSection}>
        <Logo />
      </div>
      <div style={styles.centerSection}>
        <p style={styles.emailPrompt} className="email-prompt">Please enter your email</p>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder="you@company.com"
          style={styles.underlineInput}
          className="underline-input"
        />
        {(error || serverError) && (
          <p style={styles.errorText}>{error || serverError}</p>
        )}
      </div>
      <div style={styles.bottomAction}>
        <button onClick={handleSubmit} style={styles.outlineBtn} className="outline-btn">
          SUBMIT YOUR VIDEO
        </button>
      </div>
    </div>
  );
}

function UploadingScreen({ progress }) {
  return (
    <div style={styles.gradientScreen} className="gradient-screen">
      <div style={styles.gradientOverlay} />
      <div style={styles.topSection}>
        <Logo />
      </div>
      <div style={{ ...styles.centerSection, textAlign: "center" }}>
        <div style={styles.spinner} />
        <p style={styles.uploadText}>Uploading your video&hellip;</p>
        <div style={styles.uploadBar}>
          <div style={{ ...styles.uploadFill, width: `${progress}%` }} />
        </div>
        <p style={styles.uploadPercent}>{Math.round(progress)}%</p>
      </div>
      <div style={styles.bottomAction} />
    </div>
  );
}

function SuccessScreen({ onReset }) {
  return (
    <div style={styles.gradientScreen} className="gradient-screen">
      <div style={styles.gradientOverlay} />
      <div style={styles.topSection}>
        <Logo />
      </div>
      <div style={{ ...styles.centerSection, textAlign: "center" }}>
        <div style={styles.checkCircle}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </div>
        <h2 style={styles.successTitle}>THANK YOU!</h2>
        <p style={styles.successSubtext}>Your video has been submitted successfully. Our team will review it shortly.</p>
      </div>
      <div style={styles.bottomAction}>
        <button onClick={onReset} style={styles.outlineBtn} className="outline-btn">
          RECORD ANOTHER
        </button>
      </div>
    </div>
  );
}

// ─── App ────────────────────────────────────────────────────

export default function App() {
  const [screen, setScreen] = useState("welcome");
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState("");

  const handleVideoReady = (blob) => {
    setRecordedBlob(blob);
    setScreen("email");
  };

  const handleEmail = (emailValue) => {
    setScreen("uploading");
    uploadVideo(recordedBlob, emailValue);
  };

  const uploadVideo = (blob, emailValue) => {
    setUploadProgress(0);
    setUploadError("");

    const formData = new FormData();
    formData.append("video", blob, "testimonial.webm");
    formData.append("email", emailValue);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setUploadProgress((e.loaded / e.total) * 100);
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        setUploadProgress(100);
        setTimeout(() => setScreen("success"), 400);
      } else {
        let msg = "Upload failed. Please try again.";
        try { const r = JSON.parse(xhr.responseText); if (r.error) msg = r.error; } catch (e) { /* ignore */ }
        setUploadError(msg);
        setScreen("email");
      }
    };

    xhr.onerror = () => {
      setUploadError("Network error. Please try again.");
      setScreen("email");
    };

    xhr.send(formData);
  };

  const handleReset = () => {
    setScreen("welcome");
    setRecordedBlob(null);
    setUploadProgress(0);
    setUploadError("");
  };

  return (
    <div style={styles.app}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@700&display=swap" rel="stylesheet" />
      {screen === "welcome" && <WelcomeScreen onStart={() => setScreen("record")} />}
      {screen === "record" && <RecordScreen onNext={handleVideoReady} onBack={() => setScreen("welcome")} />}
      {screen === "email" && <EmailScreen onNext={handleEmail} onBack={() => setScreen("record")} error={uploadError} />}
      {screen === "uploading" && <UploadingScreen progress={uploadProgress} />}
      {screen === "success" && <SuccessScreen onReset={handleReset} />}
    </div>
  );
}

// ─── Styles ─────────────────────────────────────────────────

const styles = {
  app: {
    fontFamily: "'DM Sans', sans-serif",
    color: "#fff",
    minHeight: "100vh",
    background: "#050510",
  },

  // ─── Gradient screens (welcome, email, upload, success)
  gradientScreen: {
    minHeight: "100vh",
    background: GRADIENT_BG,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "48px 32px",
    boxSizing: "border-box",
    position: "relative",
    overflow: "hidden",
  },
  gradientOverlay: {
    position: "absolute",
    inset: 0,
    background: "radial-gradient(circle at 30% 30%, rgba(100,50,200,0.12) 0%, transparent 60%)",
    pointerEvents: "none",
  },
  topSection: {
    position: "relative",
    zIndex: 1,
    flexShrink: 0,
    paddingTop: 16,
  },
  centerSection: {
    position: "relative",
    zIndex: 1,
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    maxWidth: 400,
  },
  bottomAction: {
    position: "relative",
    zIndex: 1,
    flexShrink: 0,
    paddingBottom: 16,
  },

  // ─── Logo
  logo: {
    marginBottom: 0,
  },
  logoText: {
    fontFamily: "'Space Mono', monospace",
    fontSize: 22,
    fontWeight: 700,
    color: "#fff",
    letterSpacing: "0.06em",
  },

  // ─── Welcome
  welcomeHeading: {
    fontSize: 32,
    fontWeight: 700,
    textAlign: "center",
    lineHeight: 1.25,
    letterSpacing: "0.03em",
    textTransform: "uppercase",
    margin: 0,
    maxWidth: 360,
  },

  // ─── Buttons
  outlineBtn: {
    padding: "16px 44px",
    background: "transparent",
    border: "2px solid rgba(255,255,255,0.55)",
    borderRadius: 4,
    color: "#fff",
    fontSize: 12,
    fontWeight: 700,
    fontFamily: "'DM Sans', sans-serif",
    letterSpacing: "0.18em",
    textTransform: "uppercase",
    cursor: "pointer",
    transition: "all 0.2s",
    whiteSpace: "nowrap",
  },
  filledBtn: {
    padding: "16px 44px",
    background: "rgba(194,24,91,0.8)",
    border: "2px solid rgba(194,24,91,0.9)",
    borderRadius: 4,
    color: "#fff",
    fontSize: 12,
    fontWeight: 700,
    fontFamily: "'DM Sans', sans-serif",
    letterSpacing: "0.18em",
    textTransform: "uppercase",
    cursor: "pointer",
    transition: "all 0.2s",
    whiteSpace: "nowrap",
  },

  // ─── Back button
  backBtn: {
    position: "absolute",
    top: 16,
    left: 16,
    zIndex: 10,
    width: 40,
    height: 40,
    borderRadius: "50%",
    background: "rgba(0,0,0,0.45)",
    border: "none",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    backdropFilter: "blur(8px)",
    padding: 0,
    transition: "all 0.2s",
  },
  backBtnGradient: {
    position: "absolute",
    top: 16,
    left: 16,
    zIndex: 10,
    width: 40,
    height: 40,
    borderRadius: "50%",
    background: "rgba(255,255,255,0.1)",
    border: "none",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    backdropFilter: "blur(8px)",
    padding: 0,
    transition: "all 0.2s",
  },

  // ─── Camera screen
  cameraScreen: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    background: "#0a0e1a",
    overflow: "hidden",
    position: "relative",
  },
  cameraView: {
    flex: 1,
    position: "relative",
    overflow: "hidden",
    background: "#000",
  },
  hiddenVideo: {
    position: "absolute",
    width: 1,
    height: 1,
    opacity: 0,
    pointerEvents: "none",
  },
  cameraFeed: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    display: "block",
  },
  cameraErrorOverlay: {
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 32,
    color: "rgba(255,255,255,0.5)",
    fontSize: 15,
    textAlign: "center",
  },

  // ─── Recording indicator
  recIndicator: {
    position: "absolute",
    top: 20,
    left: 20,
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 16px",
    background: "rgba(0,0,0,0.55)",
    borderRadius: 24,
    backdropFilter: "blur(8px)",
  },
  recDot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    background: "#e53935",
    animation: "pulse 1s infinite",
  },
  recText: {
    fontSize: 13,
    fontWeight: 600,
    fontFamily: "'Space Mono', monospace",
    color: "#fff",
  },

  // ─── Countdown
  countdownOverlay: {
    position: "absolute",
    inset: 0,
    background: "rgba(0,0,0,0.5)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  countdownNum: {
    fontSize: 96,
    fontWeight: 700,
    fontFamily: "'Space Mono', monospace",
    color: "#fff",
    textShadow: "0 0 60px rgba(194,24,91,0.5)",
  },

  // ─── Progress bar
  progressBar: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    height: 3,
    background: "rgba(255,255,255,0.15)",
  },
  progressFill: {
    height: "100%",
    background: "#e53935",
    transition: "width 1s linear",
  },

  // ─── Bottom panel (camera screen)
  bottomPanel: {
    background: "#0a0e1a",
    padding: "16px 24px 32px",
    flexShrink: 0,
  },

  // ─── Background picker
  bgSection: {
    marginBottom: 20,
  },
  bgTitle: {
    fontSize: 13,
    fontWeight: 500,
    color: "rgba(255,255,255,0.5)",
    textAlign: "center",
    margin: "0 0 12px 0",
    letterSpacing: "0.03em",
  },
  bgThumbs: {
    display: "flex",
    gap: 10,
    justifyContent: "center",
    flexWrap: "wrap",
  },
  bgThumb: {
    width: 64,
    height: 44,
    borderRadius: 8,
    border: "2px solid transparent",
    cursor: "pointer",
    transition: "all 0.15s",
    padding: 0,
    flexShrink: 0,
  },
  bgThumbActive: {
    border: "2px solid #fff",
    boxShadow: "0 0 12px rgba(255,255,255,0.25)",
  },

  // ─── Controls
  controlsRow: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  recordBtn: {
    width: 72,
    height: 72,
    borderRadius: "50%",
    background: "transparent",
    border: "4px solid rgba(255,255,255,0.65)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    padding: 0,
    transition: "all 0.15s",
  },
  recordDot: {
    width: 52,
    height: 52,
    borderRadius: "50%",
    background: "#e53935",
    transition: "all 0.15s",
  },
  stopSquare: {
    width: 28,
    height: 28,
    borderRadius: 4,
    background: "#e53935",
  },
  previewBtns: {
    display: "flex",
    gap: 16,
    justifyContent: "center",
    width: "100%",
  },

  // ─── Email screen
  emailPrompt: {
    fontSize: 18,
    fontWeight: 400,
    color: "rgba(255,255,255,0.75)",
    marginBottom: 8,
    textAlign: "center",
    letterSpacing: "0.02em",
  },
  underlineInput: {
    width: "100%",
    maxWidth: 300,
    padding: "14px 4px",
    background: "transparent",
    border: "none",
    borderBottom: "1px solid rgba(255,255,255,0.35)",
    color: "#fff",
    fontSize: 16,
    fontFamily: "'DM Sans', sans-serif",
    textAlign: "center",
    outline: "none",
    transition: "border-color 0.2s",
    boxSizing: "border-box",
  },
  errorText: {
    color: "#ff6b6b",
    fontSize: 13,
    marginTop: 12,
    textAlign: "center",
  },

  // ─── Upload screen
  spinner: {
    width: 48,
    height: 48,
    border: "3px solid rgba(255,255,255,0.12)",
    borderTop: "3px solid #fff",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
    margin: "0 auto 28px",
  },
  uploadText: {
    fontSize: 16,
    color: "rgba(255,255,255,0.65)",
    margin: "0 0 20px 0",
  },
  uploadBar: {
    width: "100%",
    maxWidth: 260,
    height: 3,
    background: "rgba(255,255,255,0.12)",
    borderRadius: 2,
    overflow: "hidden",
  },
  uploadFill: {
    height: "100%",
    background: "#fff",
    transition: "width 0.3s",
    borderRadius: 2,
  },
  uploadPercent: {
    fontSize: 14,
    color: "rgba(255,255,255,0.4)",
    marginTop: 10,
  },

  // ─── Success screen
  checkCircle: {
    width: 80,
    height: 80,
    borderRadius: "50%",
    background: "rgba(255,255,255,0.12)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 28,
  },
  successTitle: {
    fontSize: 24,
    fontWeight: 700,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    margin: "0 0 12px 0",
  },
  successSubtext: {
    fontSize: 15,
    color: "rgba(255,255,255,0.55)",
    lineHeight: 1.6,
    textAlign: "center",
    maxWidth: 300,
    margin: 0,
  },
};

// Inject keyframes
if (typeof document !== "undefined") {
  const styleEl = document.createElement("style");
  styleEl.textContent = `
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
    @keyframes spin { to{transform:rotate(360deg)} }
    input:focus { border-bottom-color: rgba(255,255,255,0.8) !important; }
    button:hover { opacity: 0.88; }
    button:active { transform: scale(0.97); }
    .record-btn:hover { border-color: rgba(255,255,255,0.9) !important; }
    .outline-btn:hover { border-color: rgba(255,255,255,0.85) !important; }
  `;
  document.head.appendChild(styleEl);
}
