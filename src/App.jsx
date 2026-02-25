import { useState, useRef, useEffect, useCallback } from "react";

const GRADIENT_BG = "linear-gradient(155deg, #0a1628 0%, #162052 20%, #3b1760 45%, #7b1a5e 70%, #c2185b 95%)";

// ─── MediaPipe Configuration ──────────────────────────────────
const MEDIAPIPE_WASM_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm";
const SELFIE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite";

// Landscape model native input resolution — 256×144 (16:9).
// Running segmentation at native resolution avoids internal rescaling;
// the joint bilateral shader upsamples to full video resolution in the GPU.
const SEG_WIDTH = 256;
const SEG_HEIGHT = 144;

const BACKGROUNDS = [
  { id: "none", label: "None", type: "none", preview: "#ffffff" },
  { id: "living-room", label: "Living Room", type: "image", src: "/backgrounds/living-room.jpg", preview: "linear-gradient(135deg, #c8956a, #f0ebe0)" },
  { id: "home-office", label: "Home Office", type: "image", src: "/backgrounds/home-office.jpg", preview: "linear-gradient(135deg, #7a9ab0, #e8ecf0)" },
  { id: "library", label: "Library", type: "image", src: "/backgrounds/library.jpg", preview: "linear-gradient(135deg, #7a5c3c, #c8a050)" },
  { id: "upload", label: "Custom", type: "upload", preview: "linear-gradient(135deg, #333, #666)" },
];

const MAX_DURATION = 30;

// ─── MediaPipe Segmenter Hook ─────────────────────────────────
function useSegmenter() {
  const segmenterRef = useRef(null);
  const [segmenterReady, setSegmenterReady] = useState(false);
  const [segmenterError, setSegmenterError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const { FilesetResolver, ImageSegmenter } = await import("@mediapipe/tasks-vision");
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_CDN);

        let segmenter;
        try {
          segmenter = await ImageSegmenter.createFromOptions(vision, {
            baseOptions: { modelAssetPath: SELFIE_MODEL_URL, delegate: "GPU" },
            runningMode: "IMAGE", // IMAGE mode: raw per-frame output, no internal temporal lag
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          });
        } catch {
          segmenter = await ImageSegmenter.createFromOptions(vision, {
            baseOptions: { modelAssetPath: SELFIE_MODEL_URL, delegate: "CPU" },
            runningMode: "IMAGE",
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          });
        }

        if (cancelled) { segmenter.close(); return; }
        segmenterRef.current = segmenter;
        setSegmenterReady(true);
      } catch (err) {
        console.error("Failed to initialize MediaPipe segmenter:", err);
        if (!cancelled) setSegmenterError(err.message || "Segmenter failed to load");
      }
    }

    init();
    return () => {
      cancelled = true;
      if (segmenterRef.current) { segmenterRef.current.close(); segmenterRef.current = null; }
    };
  }, []);

  return { segmenterRef, segmenterReady, segmenterError };
}

// ─── WebGL2 Compositing Renderer ──────────────────────────────

const VERT_SRC = `#version 300 es
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
void main() {
  gl_Position = vec4(a_pos, 0.0, 1.0);
  v_uv = a_uv;
}`;

const FRAG_SRC = `#version 300 es
precision highp float;
uniform sampler2D u_video;
uniform sampler2D u_mask;     // EMA-smoothed mask  — stable bg/fg, no flicker
uniform sampler2D u_rawMask;  // current-frame mask — zero temporal lag
uniform sampler2D u_bg;
uniform int u_mode;
uniform vec2 u_texelSize;     // full-res video texel (1/W, 1/H) — for Sobel
uniform vec2 u_maskTexelSize; // mask-space texel  (1/256, 1/144)
in vec2 v_uv;
out vec4 fragColor;

void main() {
  vec4 vid = texture(u_video, v_uv);
  if (u_mode == 0) { fragColor = vid; return; }

  const vec3 luma = vec3(0.299, 0.587, 0.114);

  // ── Pass 1: Joint Bilateral Upsampling ────────────────────────────────────
  float lumC  = dot(vid.rgb, luma);
  float totalW = 0.0;
  float mRaw = 0.0, mBlend = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2  mUV   = v_uv + vec2(float(dx), float(dy)) * u_maskTexelSize;
      float rawV  = texture(u_rawMask, mUV).r;
      float blndV = texture(u_mask,    mUV).r;
      float lumN  = dot(texture(u_video, mUV).rgb, luma);
      float wS = exp(-float(dx*dx + dy*dy) * 0.5);
      float dL = lumN - lumC;
      float wR = exp(-dL * dL * 35.0);
      float w  = wS * wR;
      mRaw   += rawV  * w;
      mBlend += blndV * w;
      totalW += w;
    }
  }
  mRaw   /= totalW;
  mBlend /= totalW;

  // ── Pass 2: Boundary-aware raw / blend mix ────────────────────────────────
  float uncertainty = 1.0 - abs(mRaw * 2.0 - 1.0);
  float m = mix(mBlend, mRaw, clamp(uncertainty * 2.0, 0.0, 1.0));

  // ── Pass 3: Full-resolution video-space edge snap (Sobel) ─────────────────
  float lumR = dot(texture(u_video, v_uv + vec2( u_texelSize.x, 0.0)).rgb, luma);
  float lumL = dot(texture(u_video, v_uv + vec2(-u_texelSize.x, 0.0)).rgb, luma);
  float lumU = dot(texture(u_video, v_uv + vec2(0.0,  u_texelSize.y)).rgb, luma);
  float lumD = dot(texture(u_video, v_uv + vec2(0.0, -u_texelSize.y)).rgb, luma);
  float videoEdge = clamp(length(vec2(lumR - lumL, lumU - lumD)) * 7.0, 0.0, 1.0);
  float mSnap = step(0.5, m);
  m = mix(m, mSnap, videoEdge * uncertainty * 0.85);

  // ── Final feather ─────────────────────────────────────────────────────────
  m = smoothstep(0.33, 0.67, m);

  vec4 bg = texture(u_bg, v_uv);
  fragColor = mix(bg, vid, m);
}`;

function compileShader(gl, src, type) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(s));
  }
  return s;
}

function createGLTexture(gl, filter) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  return tex;
}

function initWebGL(gl) {
  const hasFloatLinear = gl.getExtension("OES_texture_float_linear");
  const maskFilter = hasFloatLinear ? gl.LINEAR : gl.NEAREST;

  const vs = compileShader(gl, VERT_SRC, gl.VERTEX_SHADER);
  const fs = compileShader(gl, FRAG_SRC, gl.FRAGMENT_SHADER);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  const verts = new Float32Array([
    -1, -1, 0, 1,
     1, -1, 1, 1,
    -1,  1, 0, 0,
     1,  1, 1, 0,
  ]);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, "a_pos");
  const aUv = gl.getAttribLocation(program, "a_uv");
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
  gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 16, 8);
  gl.enableVertexAttribArray(aPos);
  gl.enableVertexAttribArray(aUv);
  gl.bindVertexArray(null);

  const videoTex = createGLTexture(gl, gl.LINEAR);
  const maskTex = createGLTexture(gl, maskFilter);
  const rawMaskTex = createGLTexture(gl, maskFilter);
  const bgTex = createGLTexture(gl, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([26, 26, 46, 255]));

  gl.useProgram(program);
  const uniforms = {
    u_video: gl.getUniformLocation(program, "u_video"),
    u_mask: gl.getUniformLocation(program, "u_mask"),
    u_rawMask: gl.getUniformLocation(program, "u_rawMask"),
    u_bg: gl.getUniformLocation(program, "u_bg"),
    u_mode: gl.getUniformLocation(program, "u_mode"),
    u_texelSize: gl.getUniformLocation(program, "u_texelSize"),
  };
  gl.uniform1i(uniforms.u_video, 0);
  gl.uniform1i(uniforms.u_mask, 1);
  gl.uniform1i(uniforms.u_bg, 2);
  gl.uniform1i(uniforms.u_rawMask, 3);

  const u_maskTexelSizeLoc = gl.getUniformLocation(program, "u_maskTexelSize");
  gl.uniform2f(u_maskTexelSizeLoc, 1.0 / SEG_WIDTH, 1.0 / SEG_HEIGHT);

  return { program, vao, buf, textures: { video: videoTex, mask: maskTex, rawMask: rawMaskTex, bg: bgTex }, uniforms };
}

// ─── WebGL background compositing with ML segmentation ────────
function useBackgroundEffect(videoRef, canvasRef, selectedBg, segmenterRef, segmenterReady, uploadedImage, bgImagesRef) {
  const selectedBgRef = useRef(selectedBg);
  const segmenterReadyRef = useRef(segmenterReady);
  const uploadedImageRef = useRef(uploadedImage);
  selectedBgRef.current = selectedBg;
  segmenterReadyRef.current = segmenterReady;
  uploadedImageRef.current = uploadedImage;

  const rendererRef = useRef(null);
  const blurCanvasRef = useRef(null);
  const blurCtxRef = useRef(null);
  const animFrameRef = useRef(null);
  const lastDimsRef = useRef({ w: 0, h: 0 });
  const lastBgKeyRef = useRef(null);
  const frameCountRef = useRef(0);
  const maskAllocRef = useRef({ w: 0, h: 0 });
  const hasMaskRef = useRef(false);
  const segCanvasRef = useRef(null);
  const segCtxRef = useRef(null);
  const blendMaskRef = useRef(null);
  const avgFrameTimeRef = useRef(16);
  const skipIntervalRef = useRef(1);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false, alpha: false });
    if (!gl) { console.error("WebGL2 not supported"); return; }
    rendererRef.current = initWebGL(gl);

    blurCanvasRef.current = document.createElement("canvas");
    blurCtxRef.current = blurCanvasRef.current.getContext("2d");

    const segCanvas = document.createElement("canvas");
    segCanvas.width = SEG_WIDTH;
    segCanvas.height = SEG_HEIGHT;
    segCanvasRef.current = segCanvas;
    segCtxRef.current = segCanvas.getContext("2d");

    const draw = () => {
      const video = videoRef.current;
      if (!gl || !video || video.paused || video.ended) {
        animFrameRef.current = requestAnimationFrame(draw);
        return;
      }

      const w = video.videoWidth || 640;
      const h = video.videoHeight || 480;
      const r = rendererRef.current;
      const curBg = selectedBgRef.current;
      const curReady = segmenterReadyRef.current;
      const curUploaded = uploadedImageRef.current;

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, r.textures.video);
      if (lastDimsRef.current.w !== w || lastDimsRef.current.h !== h) {
        canvas.width = w;
        canvas.height = h;
        gl.viewport(0, 0, w, h);
        gl.useProgram(r.program);
        gl.uniform2f(r.uniforms.u_texelSize, 1.0 / w, 1.0 / h);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        lastDimsRef.current = { w, h };
      }

      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, video);

      if (curBg === "none" || !curReady || !segmenterRef.current) {
        gl.useProgram(r.program);
        gl.uniform1i(r.uniforms.u_mode, 0);
        gl.bindVertexArray(r.vao);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        animFrameRef.current = requestAnimationFrame(draw);
        return;
      }

      frameCountRef.current++;
      const shouldSegment = frameCountRef.current % skipIntervalRef.current === 0;

      if (shouldSegment) {
        const segStart = performance.now();

        segCtxRef.current.drawImage(video, 0, 0, SEG_WIDTH, SEG_HEIGHT);

        let mask = null;
        try {
          const result = segmenterRef.current.segment(segCanvasRef.current);
          if (result.confidenceMasks?.length > 0) {
            mask = result.confidenceMasks[0].getAsFloat32Array();
          }
        } catch { /* fall through */ }

        if (mask) {
          gl.activeTexture(gl.TEXTURE3);
          gl.bindTexture(gl.TEXTURE_2D, r.textures.rawMask);
          if (maskAllocRef.current.w !== SEG_WIDTH || maskAllocRef.current.h !== SEG_HEIGHT) {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, SEG_WIDTH, SEG_HEIGHT, 0, gl.RED, gl.FLOAT, mask);
          } else {
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SEG_WIDTH, SEG_HEIGHT, gl.RED, gl.FLOAT, mask);
          }

          const prev = blendMaskRef.current;
          const hasPrev = prev && prev.length === mask.length;
          if (!hasPrev) {
            blendMaskRef.current = new Float32Array(mask);
          } else {
            let diff = 0;
            for (let i = 0; i < mask.length; i += 16) {
              diff += Math.abs(mask[i] - prev[i]);
            }
            diff /= (mask.length >> 4);
            const globalAlpha = Math.min(0.9, 0.15 + diff * 6.0);
            const blend = blendMaskRef.current;
            for (let i = 0; i < mask.length; i++) {
              const curr = mask[i];
              const uncertainty = 1.0 - Math.abs(curr * 2.0 - 1.0);
              const localAlpha = globalAlpha + uncertainty * (1.0 - globalAlpha);
              blend[i] = curr * localAlpha + blend[i] * (1.0 - localAlpha);
            }
          }

          gl.activeTexture(gl.TEXTURE1);
          gl.bindTexture(gl.TEXTURE_2D, r.textures.mask);
          if (maskAllocRef.current.w !== SEG_WIDTH || maskAllocRef.current.h !== SEG_HEIGHT) {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, SEG_WIDTH, SEG_HEIGHT, 0, gl.RED, gl.FLOAT, blendMaskRef.current);
            maskAllocRef.current = { w: SEG_WIDTH, h: SEG_HEIGHT };
          } else {
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SEG_WIDTH, SEG_HEIGHT, gl.RED, gl.FLOAT, blendMaskRef.current);
          }
          hasMaskRef.current = true;
        }

        const segTime = performance.now() - segStart;
        const avg = avgFrameTimeRef.current = avgFrameTimeRef.current * 0.9 + segTime * 0.1;
        if (avg > 8 && skipIntervalRef.current < 2) skipIntervalRef.current++;
        else if (avg < 3 && skipIntervalRef.current > 1) skipIntervalRef.current--;
      }

      if (!hasMaskRef.current) {
        gl.useProgram(r.program);
        gl.uniform1i(r.uniforms.u_mode, 0);
        gl.bindVertexArray(r.vao);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        animFrameRef.current = requestAnimationFrame(draw);
        return;
      }

      const bg = BACKGROUNDS.find((b) => b.id === curBg);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, r.textures.bg);

      if (bg?.type === "blur") {
        const bw = Math.round(w / 2), bh = Math.round(h / 2);
        const bc = blurCanvasRef.current, bctx = blurCtxRef.current;
        if (bc.width !== bw || bc.height !== bh) { bc.width = bw; bc.height = bh; }
        bctx.filter = `blur(${Math.max(1, Math.round(bg.blurPx / 2))}px)`;
        bctx.drawImage(video, 0, 0, bw, bh);
        bctx.filter = "none";
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bc);
        lastBgKeyRef.current = null;
      } else if (bg?.type === "image") {
        const img = bgImagesRef?.current?.[bg.id];
        if (img && lastBgKeyRef.current !== bg.id) {
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
          lastBgKeyRef.current = bg.id;
        }
      } else if (bg?.type === "upload" && curUploaded) {
        const key = "upload:" + curUploaded.src;
        if (lastBgKeyRef.current !== key) {
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, curUploaded);
          lastBgKeyRef.current = key;
        }
      }

      gl.useProgram(r.program);
      gl.uniform1i(r.uniforms.u_mode, 1);
      gl.bindVertexArray(r.vao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      animFrameRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (rendererRef.current && gl) {
        gl.deleteProgram(rendererRef.current.program);
        gl.deleteBuffer(rendererRef.current.buf);
        Object.values(rendererRef.current.textures).forEach((t) => gl.deleteTexture(t));
        gl.deleteVertexArray(rendererRef.current.vao);
        rendererRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoRef, canvasRef]);
}

// ─── Shared Components ────────────────────────────────────────

function Logo({ onClick }) {
  return (
    <div className="logo" style={{ ...styles.logo, ...(onClick ? { cursor: "pointer" } : {}) }} onClick={onClick}>
      <div style={styles.logoIcon}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="23 7 16 12 23 17 23 7" />
          <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
        </svg>
      </div>
      <span style={styles.logoText}>VideoVoice</span>
    </div>
  );
}

// ─── Screens ─────────────────────────────────────────────────

function WelcomeScreen({ onStart }) {
  return (
    <div style={styles.gradientScreen} className="gradient-screen">
      <div style={styles.gradientOverlay} />

      {/* Nav bar */}
      <header style={styles.navBar} className="anim-fade-in">
        <Logo />
      </header>

      {/* Hero */}
      <div style={styles.heroSection}>
        {/* Left column */}
        <div style={styles.heroLeft}>

          <h1 style={styles.welcomeHeading} className="anim-slide-up d2 welcome-heading">
            Why do you love<br />working here?
          </h1>

          <p style={styles.welcomeSub} className="anim-slide-up d3">
            Share your story in 30 seconds. Record, review, and submit — no app needed.
          </p>

          <div style={styles.heroSteps} className="anim-slide-up d4">
            {[
              { n: "1", label: "Record", desc: "Answer on camera" },
              { n: "2", label: "Review", desc: "Keep or retake" },
              { n: "3", label: "Submit", desc: "Enter email & send" },
            ].map((step) => (
              <div key={step.n} style={styles.heroStep}>
                <div style={styles.heroStepNum}>{step.n}</div>
                <div>
                  <div style={styles.heroStepLabel}>{step.label}</div>
                  <div style={styles.heroStepDesc}>{step.desc}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="anim-slide-up d5" style={{ display: "flex", justifyContent: "center", marginTop: 60 }}>
            <button onClick={onStart} style={styles.filledBtn} className="filled-btn start-btn">
              <span style={styles.startBtnDot} />
              Record Your Answer
              <svg style={{ marginLeft: 10, flexShrink: 0 }} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M15 10l4.553-2.069A1 1 0 0121 8.868V15.13a1 1 0 01-1.447.899L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
              </svg>
            </button>
          </div>
        </div>

        {/* Right column — decorative card (hidden on small screens via CSS) */}
        <div style={styles.heroRight} className="hero-right anim-scale-in d3">
          <div style={styles.heroCard}>
            <div style={styles.heroCardBadge}>
              <span style={{ ...styles.heroBadgeDot, background: "#e53935" }} />
              <span style={{ fontSize: 12, fontWeight: 600, color: "#fff" }}>LIVE</span>
            </div>
            <div style={styles.heroCardCamera}>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M15 10l4.553-2.069A1 1 0 0121 8.868V15.13a1 1 0 01-1.447.899L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
              </svg>
              <p style={{ color: "rgba(255,255,255,0.3)", fontSize: 13, margin: "12px 0 0", textAlign: "center" }}>
                Your camera will appear here
              </p>
            </div>
            <div style={styles.heroCardControls}>
              <div style={styles.heroCardRecBtn} className="hero-rec-btn" />
            </div>
          </div>
        </div>
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
  const fileInputRef = useRef(null);

  const [phase, setPhase] = useState("setup");
  const [countdown, setCountdown] = useState(3);
  const [elapsed, setElapsed] = useState(0);
  const [selectedBg, setSelectedBg] = useState("none");
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);

  const bgImagesRef = useRef({});
  const { segmenterRef, segmenterReady, segmenterError } = useSegmenter();
  useBackgroundEffect(videoRef, canvasRef, selectedBg, segmenterRef, segmenterReady, uploadedImage, bgImagesRef);

  useEffect(() => {
    BACKGROUNDS.filter((bg) => bg.type === "image" && bg.src).forEach((bg) => {
      const img = new Image();
      img.onload = () => { bgImagesRef.current[bg.id] = img; };
      img.src = bg.src;
    });
  }, []);

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
  }, []); // eslint-disable-line

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
  }, []); // eslint-disable-line

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

  const handleImageUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file || !file.type.startsWith("image/")) return;
    if (uploadedImage?.src?.startsWith("blob:")) {
      URL.revokeObjectURL(uploadedImage.src);
    }
    const img = new Image();
    img.onload = () => {
      setUploadedImage(img);
      setSelectedBg("upload");
    };
    img.src = URL.createObjectURL(file);
  }, [uploadedImage]);

  const progress = (elapsed / MAX_DURATION) * 100;
  const timeLeft = MAX_DURATION - elapsed;

  return (
    <div style={styles.cameraScreen} className="camera-screen">

      {/* Hidden source video */}
      <video ref={videoRef} style={styles.hiddenVideo} muted playsInline />

      {/* Camera / preview canvas */}
      <div style={styles.cameraView} className="camera-view">
        <canvas ref={canvasRef} style={{ ...styles.cameraFeed, display: phase === "preview" ? "none" : "block" }} />
        {phase === "preview" && (
          <video src={recordedUrl} style={styles.cameraFeed} controls autoPlay loop />
        )}

        {cameraError && (
          <div style={styles.cameraErrorOverlay}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: 12 }}>
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>{cameraError}</span>
          </div>
        )}

        {/* AI loading badge */}
        {!segmenterReady && !segmenterError && selectedBg !== "none" && phase !== "preview" && (
          <div style={styles.segmenterLoadingOverlay} className="anim-fade-in">
            <span style={styles.segmenterLoadingDot} />
            <span style={styles.segmenterLoadingLabel}>Loading AI…</span>
          </div>
        )}

        {/* Countdown */}
        {phase === "countdown" && (
          <div style={styles.countdownOverlay}>
            <span key={countdown} style={styles.countdownNum} className="countdown-num">{countdown}</span>
          </div>
        )}

        {/* Recording indicator + timer */}
        {phase === "recording" && (
          <>
            <div style={styles.recIndicator} className="anim-slide-down">
              <span style={styles.recDot} className="rec-dot" />
              <span style={styles.recText}>REC</span>
              <span style={styles.recTime}>{elapsed}s / {MAX_DURATION}s</span>
            </div>
            {/* Countdown ring top-right */}
            <div style={styles.recRing} className="anim-fade-in">
              <svg viewBox="0 0 44 44" style={{ width: 44, height: 44, transform: "rotate(-90deg)" }}>
                <circle cx="22" cy="22" r="18" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="3.5" />
                <circle cx="22" cy="22" r="18" fill="none"
                  stroke={timeLeft <= 10 ? "#e53935" : "#fff"} strokeWidth="3.5"
                  strokeDasharray={`${2 * Math.PI * 18}`}
                  strokeDashoffset={`${2 * Math.PI * 18 * (elapsed / MAX_DURATION)}`}
                  strokeLinecap="round"
                  style={{ transition: "stroke-dashoffset 1s linear, stroke 0.3s" }}
                />
              </svg>
              <span style={{ position: "absolute", fontSize: 11, fontFamily: "'Space Mono', monospace", fontWeight: 700, color: timeLeft <= 10 ? "#e53935" : "#fff" }}>
                {timeLeft}
              </span>
            </div>
          </>
        )}

        {/* Progress bar — recording */}
        {phase === "recording" && (
          <div style={styles.progressBar}>
            <div style={{ ...styles.progressFill, width: `${progress}%` }} />
          </div>
        )}

        {/* ── Film set HUD (setup phase) */}
        {phase === "setup" && (
          <div style={styles.filmHUD} className="anim-fade-in">
            <div style={styles.filmHUDLeft}>
              <Logo onClick={onBack} />
            </div>
            <div style={styles.standbyBadge}>
              <span style={styles.standbyDot} className="standby-dot" />
              STANDBY
            </div>
            <span style={styles.filmReadout}>29.97 fps<br />1280×720</span>
          </div>
        )}

        {/* Viewfinder corners */}
        {phase === "setup" && cameraReady && (
          <>
            <div style={styles.fcTL} className="anim-fade-in" />
            <div style={styles.fcTR} className="anim-fade-in" />
            <div style={styles.fcBL} className="anim-fade-in" />
            <div style={styles.fcBR} className="anim-fade-in" />
          </>
        )}
      </div>

      {/* Bottom panel */}
      <div style={styles.bottomPanel} className="bottom-panel">

        {/* Background picker */}
        {phase === "setup" && (
          <div style={styles.bgSection} className="bg-section anim-slide-up">
            <p style={styles.bgTitle}>
              Virtual Background
              {!segmenterReady && !segmenterError && (
                <span style={styles.segmenterLoadingText}> · loading AI…</span>
              )}
            </p>
            <div style={styles.bgThumbs} className="bg-thumbs">
              {BACKGROUNDS.map((bg) => (
                <button
                  key={bg.id}
                  onClick={() => {
                    if (bg.type === "upload") {
                      fileInputRef.current?.click();
                    } else {
                      setSelectedBg(bg.id);
                    }
                  }}
                  style={{
                    ...styles.bgThumb,
                    background: bg.id === "upload" && uploadedImage
                      ? `url(${uploadedImage.src}) center/cover`
                      : bg.type === "image" && bg.src
                      ? `url(${bg.src}) center/cover`
                      : bg.preview,
                    ...(selectedBg === bg.id ? styles.bgThumbActive : {}),
                  }}
                  className="bg-thumb"
                  title={bg.label}
                >
                  {bg.type === "none" && (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2.5" strokeLinecap="round">
                      <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  )}
                  {bg.type === "upload" && !uploadedImage && (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                  )}
                  <span style={styles.bgThumbLabel}>{bg.label}</span>
                </button>
              ))}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: "none" }}
            />
          </div>
        )}

        {/* Controls row */}
        <div style={styles.controlsRow} className="controls-row">
          {phase === "setup" && (
            <button
              onClick={startCountdown}
              disabled={!cameraReady}
              style={{ ...styles.recordBtn, opacity: cameraReady ? 1 : 0.4 }}
              className="record-btn"
              aria-label="Start recording"
            >
              <span style={styles.recordDot} />
            </button>
          )}
          {phase === "countdown" && (
            <button style={{ ...styles.recordBtn, opacity: 0.4 }} disabled aria-label="Preparing…">
              <span style={styles.recordDot} />
            </button>
          )}
          {phase === "recording" && (
            <button onClick={stopRecording} style={{ ...styles.recordBtn, ...styles.recordBtnActive }} className="record-btn" aria-label="Stop recording">
              <span style={styles.stopSquare} />
            </button>
          )}
          {phase === "preview" && (
            <div style={styles.previewBtns} className="preview-btns anim-slide-up">
              <button onClick={retake} style={styles.outlineBtn} className="outline-btn">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 7 }}>
                  <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 102.13-9.36L1 10" />
                </svg>
                Retake
              </button>
              <button onClick={() => onNext(recordedBlob)} style={styles.filledBtn} className="filled-btn">
                Use This
                <svg style={{ marginLeft: 8 }} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" />
                </svg>
              </button>
            </div>
          )}
        </div>

        {/* Hint */}
        {phase === "setup" && cameraReady && (
          <p style={styles.hint} className="anim-fade-in d5">
            Tap to start · max {MAX_DURATION} seconds
          </p>
        )}
      </div>
    </div>
  );
}

function EmailScreen({ onNext, onBack, onHome, error: serverError }) {
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [focused, setFocused] = useState(false);

  const handleSubmit = () => {
    if (!email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      setError("Please enter a valid email address");
      return;
    }
    setError("");
    onNext(email);
  };

  return (
    <div style={styles.gradientScreen} className="gradient-screen">
      <div style={styles.gradientOverlay} />

      {/* Nav */}
      <div style={styles.pageHead} className="anim-fade-in">
        <Logo onClick={onHome} />
      </div>

      {/* Content */}
      <div style={styles.centerSection}>
        <div style={styles.emailCard} className="anim-slide-up d2">
          {/* Icon */}
          <div style={styles.emailIcon} className="anim-scale-in d1">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
              <polyline points="22,6 12,13 2,6" />
            </svg>
          </div>

          <h2 style={styles.emailHeading} className="anim-slide-up d2">
            Almost done!
          </h2>
          <p style={styles.emailSub} className="anim-slide-up d3">
            Where should we send your recording?
          </p>

          {/* Input */}
          <div style={styles.inputWrap} className="anim-slide-up d3">
            <label style={styles.inputLabel} htmlFor="email-input">Email address</label>
            <input
              id="email-input"
              type="email"
              value={email}
              onChange={(e) => { setEmail(e.target.value); if (error) setError(""); }}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              placeholder="you@company.com"
              style={{
                ...styles.underlineInput,
                borderColor: error
                  ? "rgba(239,68,68,0.6)"
                  : focused
                  ? "rgba(194,24,91,0.7)"
                  : "rgba(255,255,255,0.25)",
                boxShadow: focused
                  ? "0 0 0 3px rgba(194,24,91,0.15)"
                  : "none",
              }}
              className="email-input"
              autoCapitalize="off"
              autoCorrect="off"
            />
          </div>

          {/* Error */}
          {(error || serverError) && (
            <div style={styles.errorBox} className="anim-slide-up">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" style={{ flexShrink: 0 }}>
                <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {error || serverError}
            </div>
          )}

          {/* Submit */}
          <button onClick={handleSubmit} style={{ ...styles.filledBtn, ...styles.filledBtnFull }} className="filled-btn anim-slide-up d4">
            <svg style={{ marginRight: 10 }} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
            Submit Your Video
            
          </button>

          {/* Back */}
          <button onClick={onBack} style={styles.ghostBtn} className="outline-btn anim-slide-up d5">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 6 }}>
              <polyline points="15 18 9 12 15 6" />
            </svg>
            Go back
          </button>
        </div>
      </div>

      <div style={{ flexShrink: 0, height: 32 }} />
    </div>
  );
}

function UploadingScreen({ progress }) {
  const circumference = 2 * Math.PI * 28;
  return (
    <div style={styles.gradientScreen} className="gradient-screen">
      <div style={styles.gradientOverlay} />

      <header style={styles.navBar} className="anim-fade-in">
        <Logo />
      </header>

      <div style={{ ...styles.centerSection, textAlign: "center" }}>
        {/* Circular progress ring */}
        <div style={styles.progressRingWrap} className="anim-scale-in d1">
          <svg viewBox="0 0 64 64" style={{ width: 96, height: 96, transform: "rotate(-90deg)" }}>
            <circle cx="32" cy="32" r="28" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="5" />
            <circle cx="32" cy="32" r="28" fill="none" stroke="#fff" strokeWidth="5"
              strokeDasharray={circumference}
              strokeDashoffset={circumference * (1 - progress / 100)}
              strokeLinecap="round"
              style={{ transition: "stroke-dashoffset 0.4s ease" }}
            />
          </svg>
          <span style={styles.progressPct}>{Math.round(progress)}%</span>
        </div>

        <h2 style={styles.uploadTitle} className="anim-slide-up d2">Uploading…</h2>
        <p style={styles.uploadText} className="anim-slide-up d3">
          Sending your video securely
        </p>

        {/* Animated dots */}
        <div style={{ display: "flex", justifyContent: "center", gap: 7, marginTop: 20 }} className="anim-fade-in d4">
          {[0, 1, 2].map((i) => (
            <span key={i} style={{ width: 7, height: 7, borderRadius: "50%", background: "rgba(255,255,255,0.35)", display: "block", animation: `pulse 1.4s ${i * 0.22}s ease-in-out infinite` }} />
          ))}
        </div>
      </div>

      <div style={{ flexShrink: 0, height: 48 }} />
    </div>
  );
}

function SuccessScreen({ onReset }) {
  return (
    <div style={styles.gradientScreen} className="gradient-screen">
      <div style={styles.gradientOverlay} />

      <header style={styles.navBar} className="anim-fade-in">
        <Logo />
      </header>

      <div style={{ ...styles.centerSection, textAlign: "center" }}>
        {/* Animated check */}
        <div style={styles.checkCircle} className="success-circle anim-pop-in d1">
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </div>

        <h2 style={styles.successTitle} className="anim-slide-up d2">Thank you!</h2>
        <p style={styles.successSubtext} className="anim-slide-up d3">
          Your video has been submitted successfully.<br />Our team will review it shortly.
        </p>

        <div style={{ marginTop: 32 }} className="anim-slide-up d4">
          <button onClick={onReset} style={styles.outlineBtn} className="outline-btn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 8 }}>
              <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 102.13-9.36L1 10" />
            </svg>
            Record Another
          </button>
        </div>
      </div>

      <div style={{ flexShrink: 0, height: 48 }} />
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────

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
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Space+Mono:wght@700&display=swap" rel="stylesheet" />
      {screen === "welcome" && <WelcomeScreen onStart={() => setScreen("record")} />}
      {screen === "record" && <RecordScreen onNext={handleVideoReady} onBack={() => setScreen("welcome")} />}
      {screen === "email" && <EmailScreen onNext={handleEmail} onBack={() => setScreen("record")} onHome={() => setScreen("welcome")} error={uploadError} />}
      {screen === "uploading" && <UploadingScreen progress={uploadProgress} />}
      {screen === "success" && <SuccessScreen onReset={handleReset} />}
    </div>
  );
}

// ─── Styles ──────────────────────────────────────────────────

const styles = {
  // ── App shell
  app: {
    fontFamily: "'DM Sans', sans-serif",
    color: "#fff",
    minHeight: "100vh",
    background: "#07080f",
    WebkitFontSmoothing: "antialiased",
    MozOsxFontSmoothing: "grayscale",
  },

  // ── Gradient screens
  gradientScreen: {
    minHeight: "100vh",
    background: GRADIENT_BG,
    display: "flex",
    flexDirection: "column",
    position: "relative",
    overflow: "hidden",
  },
  gradientOverlay: {
    position: "absolute", inset: 0,
    background: "radial-gradient(ellipse 80% 60% at 60% 0%, rgba(120,40,220,0.18) 0%, transparent 65%)",
    pointerEvents: "none",
  },

  // ── Page head (Logo + back stacked vertically, left-aligned)
  pageHead: {
    display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 10,
    padding: "20px 28px",
    position: "relative", zIndex: 10, flexShrink: 0,
  },

  // ── Nav bar
  navBar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "20px 28px",
    position: "relative",
    zIndex: 10,
    flexShrink: 0,
  },
  navBack: {
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    width: 34, height: 34, borderRadius: 9, padding: 0, flexShrink: 0,
    background: "rgba(255,255,255,0.08)",
    border: "1px solid rgba(255,255,255,0.18)",
    color: "rgba(255,255,255,0.9)",
    cursor: "pointer", backdropFilter: "blur(10px)",
    transition: "all 0.2s",
  },

  // ── Logo
  logo: {
    display: "inline-flex",
    alignItems: "center",
    gap: 9,
    textDecoration: "none",
  },
  logoIcon: {
    width: 32, height: 32, borderRadius: 9,
    background: "linear-gradient(135deg, #c2185b, #7b1a5e)",
    display: "flex", alignItems: "center", justifyContent: "center",
    flexShrink: 0,
  },
  logoText: {
    fontFamily: "'Space Mono', monospace",
    fontSize: 18, fontWeight: 700,
    color: "#fff", letterSpacing: "0.04em",
  },

  // ── Hero (Welcome screen)
  heroSection: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    padding: "32px 28px 40px",
    gap: 48,
    position: "relative",
    zIndex: 1,
    maxWidth: 1100,
    margin: "0 auto",
    width: "100%",
    boxSizing: "border-box",
  },
  heroLeft: {
    flex: "1 1 500px",
    minWidth: 0,
  },
  heroRight: {
    flex: "0 0 320px",
    display: "none", // shown via CSS @media
  },
  heroBadge: {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    background: "rgba(255,255,255,0.07)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 20,
    padding: "6px 14px",
    marginBottom: 22,
    backdropFilter: "blur(8px)",
  },
  heroBadgeDot: {
    width: 7, height: 7, borderRadius: "50%",
    background: "#4ade80", flexShrink: 0,
    boxShadow: "0 0 6px #4ade80",
  },
  heroBadgeText: {
    fontSize: 12, fontWeight: 500,
    color: "rgba(255,255,255,0.7)",
    letterSpacing: "0.03em",
  },
  welcomeHeading: {
    fontSize: 48, fontWeight: 700, lineHeight: 1.15,
    letterSpacing: "-0.01em", margin: "0 0 20px",
    textShadow: "0 2px 24px rgba(0,0,0,0.25)",
  },
  welcomeSub: {
    fontSize: 17, color: "rgba(255,255,255,0.55)",
    lineHeight: 1.65, margin: "0 0 36px",
    maxWidth: 460,
  },
  heroSteps: {
    display: "flex",
    flexDirection: "column",
    gap: 14,
    marginBottom: 40,
  },
  heroStep: {
    display: "flex",
    alignItems: "center",
    gap: 14,
  },
  heroStepNum: {
    width: 28, height: 28, borderRadius: "50%",
    background: "rgba(194,24,91,0.25)",
    border: "1.5px solid rgba(194,24,91,0.45)",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 12, fontWeight: 700, color: "#e91e8c",
    flexShrink: 0,
  },
  heroStepLabel: {
    fontSize: 14, fontWeight: 600, color: "#fff",
  },
  heroStepDesc: {
    fontSize: 12, color: "rgba(255,255,255,0.4)",
  },

  // ── Hero decorative card
  heroCard: {
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 20,
    padding: "24px",
    backdropFilter: "blur(16px)",
    overflow: "hidden",
  },
  heroCardBadge: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    background: "rgba(0,0,0,0.3)",
    borderRadius: 6,
    padding: "4px 10px",
    marginBottom: 16,
  },
  heroCardCamera: {
    height: 160,
    background: "rgba(0,0,0,0.25)",
    borderRadius: 12,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    border: "1px dashed rgba(255,255,255,0.1)",
  },
  heroCardControls: {
    display: "flex",
    justifyContent: "center",
    paddingTop: 20,
  },
  heroCardRecBtn: {
    width: 52, height: 52, borderRadius: "50%",
    background: "transparent",
    border: "3px solid rgba(255,255,255,0.3)",
    display: "flex", alignItems: "center", justifyContent: "center",
  },

  // ── Camera screen
  cameraScreen: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    background: "#050508",
    overflow: "hidden",
    position: "relative",
  },
  cameraView: {
    position: "absolute",
    inset: 0,
    overflow: "hidden",
    background: "#000",
  },
  hiddenVideo: {
    position: "absolute",
    width: 1, height: 1,
    opacity: 0, pointerEvents: "none",
  },
  cameraFeed: {
    width: "100%", height: "100%",
    objectFit: "cover", display: "block",
  },
  cameraErrorOverlay: {
    position: "absolute", inset: 0,
    display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    padding: 32,
    color: "rgba(255,255,255,0.5)",
    fontSize: 14, textAlign: "center",
    background: "rgba(0,0,0,0.55)",
    lineHeight: 1.6,
  },

  // ── AI loading badge
  segmenterLoadingOverlay: {
    position: "absolute", top: 14, right: 14,
    display: "flex", alignItems: "center", gap: 7,
    padding: "5px 13px",
    background: "rgba(0,0,0,0.6)", borderRadius: 20,
    backdropFilter: "blur(8px)", zIndex: 5,
  },
  segmenterLoadingDot: {
    width: 7, height: 7, borderRadius: "50%",
    background: "#ffa726",
    animation: "pulse 1.4s ease-in-out infinite",
    display: "block",
  },
  segmenterLoadingLabel: {
    fontSize: 11, fontWeight: 500,
    color: "rgba(255,255,255,0.7)",
  },
  segmenterLoadingText: {
    fontSize: 11, fontWeight: 400,
    color: "rgba(255,255,255,0.35)",
  },

  // ── Recording indicator
  recIndicator: {
    position: "absolute", top: 14, left: 14,
    display: "flex", alignItems: "center", gap: 8,
    padding: "7px 14px",
    background: "rgba(0,0,0,0.6)", borderRadius: 24,
    backdropFilter: "blur(8px)", zIndex: 5,
  },
  recDot: {
    width: 8, height: 8, borderRadius: "50%",
    background: "#e53935", display: "block",
    animation: "recPulse 1s ease-in-out infinite",
  },
  recText: {
    fontSize: 11, fontWeight: 700,
    fontFamily: "'Space Mono', monospace",
    color: "#fff", letterSpacing: "0.08em",
  },
  recTime: {
    fontSize: 11,
    fontFamily: "'Space Mono', monospace",
    color: "rgba(255,255,255,0.55)",
  },
  recRing: {
    position: "absolute", top: 12, right: 12,
    width: 44, height: 44, zIndex: 5,
    display: "flex", alignItems: "center", justifyContent: "center",
  },

  // ── Countdown overlay
  countdownOverlay: {
    position: "absolute", inset: 0,
    background: "rgba(0,0,0,0.55)",
    display: "flex", alignItems: "center", justifyContent: "center",
    zIndex: 10,
  },
  countdownNum: {
    fontSize: 112, fontWeight: 700,
    fontFamily: "'Space Mono', monospace",
    color: "#fff", lineHeight: 1,
    textShadow: "0 0 80px rgba(194,24,91,0.55)",
  },

  // ── Progress bar (recording)
  progressBar: {
    position: "absolute", bottom: 0, left: 0, right: 0,
    height: 3, background: "rgba(255,255,255,0.1)",
    zIndex: 5,
  },
  progressFill: {
    height: "100%", background: "#e53935",
    transition: "width 1s linear",
  },

  // ── Back button (camera screen)
  backBtn: {
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    width: 34, height: 34, borderRadius: 9, padding: 0, flexShrink: 0,
    background: "rgba(255,255,255,0.08)",
    border: "1px solid rgba(255,255,255,0.18)",
    color: "rgba(255,255,255,0.9)",
    cursor: "pointer", backdropFilter: "blur(10px)",
    transition: "all 0.2s",
  },

  // ── Film set HUD
  filmHUD: {
    position: "absolute", top: 0, left: 0, right: 0, zIndex: 20,
    display: "flex", alignItems: "flex-start", justifyContent: "space-between",
    padding: "12px 14px",
    background: "linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, transparent 100%)",
  },
  filmHUDLeft: {
    display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 8,
  },
  standbyBadge: {
    display: "flex", alignItems: "center", gap: 6,
    background: "rgba(0,0,0,0.45)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 6, padding: "5px 12px",
    fontSize: 9, fontWeight: 700,
    color: "rgba(255,255,255,0.6)",
    letterSpacing: "0.18em",
    fontFamily: "'Space Mono', monospace",
    backdropFilter: "blur(6px)",
  },
  standbyDot: {
    width: 6, height: 6, borderRadius: "50%",
    background: "#ffb300", flexShrink: 0,
  },
  filmReadout: {
    fontSize: 9, fontWeight: 700,
    color: "rgba(255,255,255,0.38)",
    fontFamily: "'Space Mono', monospace",
    letterSpacing: "0.08em",
    textAlign: "right", lineHeight: 1.8,
  },
  // Viewfinder corners
  fcTL: { position: "absolute", top: 58, left: 16, width: 22, height: 22, zIndex: 15, borderTop: "1.5px solid rgba(255,255,255,0.45)", borderLeft: "1.5px solid rgba(255,255,255,0.45)" },
  fcTR: { position: "absolute", top: 58, right: 16, width: 22, height: 22, zIndex: 15, borderTop: "1.5px solid rgba(255,255,255,0.45)", borderRight: "1.5px solid rgba(255,255,255,0.45)" },
  fcBL: { position: "absolute", bottom: 218, left: 16, width: 22, height: 22, zIndex: 15, borderBottom: "1.5px solid rgba(255,255,255,0.45)", borderLeft: "1.5px solid rgba(255,255,255,0.45)" },
  fcBR: { position: "absolute", bottom: 218, right: 16, width: 22, height: 22, zIndex: 15, borderBottom: "1.5px solid rgba(255,255,255,0.45)", borderRight: "1.5px solid rgba(255,255,255,0.45)" },

  // ── Bottom panel (camera)
  bottomPanel: {
    position: "absolute", bottom: 0, left: 0, right: 0,
    background: "linear-gradient(to top, rgba(0,0,0,0.72) 55%, transparent 100%)",
    padding: "28px 20px 28px",
    zIndex: 10,
  },

  // ── Background picker
  bgSection: {
    marginBottom: 16,
  },
  bgTitle: {
    fontSize: 11, fontWeight: 700,
    color: "rgba(255,255,255,0.38)",
    margin: "0 0 10px",
    letterSpacing: "0.1em", textTransform: "uppercase",
  },
  bgThumbs: {
    display: "flex",
    gap: 8,
    overflowX: "auto",
    paddingBottom: 4,
    WebkitOverflowScrolling: "touch",
    scrollbarWidth: "none",
    justifyContent: "center",
  },
  bgThumb: {
    width: 56, height: 40, borderRadius: 8,
    border: "2px solid rgba(255,255,255,0.1)",
    cursor: "pointer", padding: 0, flexShrink: 0,
    display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    backgroundSize: "cover", backgroundPosition: "center",
    transition: "all 0.15s", gap: 2,
    position: "relative",
  },
  bgThumbActive: {
    border: "2.5px solid #fff",
    boxShadow: "0 0 0 3px rgba(255,255,255,0.18)",
    transform: "scale(1.06)",
  },
  bgThumbLabel: {
    fontSize: 8, color: "rgba(255,255,255,0.55)",
    letterSpacing: "0.05em", textTransform: "uppercase",
    position: "absolute", bottom: 3, left: 0, right: 0,
    textAlign: "center", fontWeight: 600,
    textShadow: "0 1px 2px rgba(0,0,0,0.8)",
  },

  // ── Controls row
  controlsRow: {
    display: "flex", justifyContent: "center", alignItems: "center",
    minHeight: 88,
  },
  recordBtn: {
    width: 76, height: 76, borderRadius: "50%",
    background: "rgba(255,255,255,0.06)",
    border: "4px solid rgba(255,255,255,0.7)",
    display: "flex", alignItems: "center", justifyContent: "center",
    cursor: "pointer", padding: 0, transition: "all 0.15s",
  },
  recordBtnActive: {
    border: "4px solid #e53935",
    background: "rgba(229,57,53,0.08)",
    animation: "recordPulse 2s ease-in-out infinite",
  },
  recordDot: {
    width: 52, height: 52, borderRadius: "50%",
    background: "#e53935", transition: "all 0.15s",
    boxShadow: "0 2px 16px rgba(229,57,53,0.5)",
  },
  stopSquare: {
    width: 28, height: 28, borderRadius: 7,
    background: "#e53935",
    boxShadow: "0 2px 14px rgba(229,57,53,0.55)",
  },
  previewBtns: {
    display: "flex", gap: 12, justifyContent: "center", width: "100%",
  },
  hint: {
    textAlign: "center", fontSize: 11,
    color: "rgba(255,255,255,0.25)",
    margin: "8px 0 0",
    letterSpacing: "0.05em",
  },

  // ── Buttons (shared)
  outlineBtn: {
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    padding: "13px 26px",
    background: "rgba(255,255,255,0.06)",
    border: "1.5px solid rgba(255,255,255,0.28)",
    borderRadius: 10,
    color: "#fff", fontSize: 14, fontWeight: 600,
    fontFamily: "'DM Sans', sans-serif",
    letterSpacing: "0.02em",
    cursor: "pointer", transition: "all 0.2s",
    whiteSpace: "nowrap", backdropFilter: "blur(8px)",
  },
  startBtnDot: {
    width: 8, height: 8, borderRadius: "50%",
    background: "#ff4d4d",
    marginRight: 10, flexShrink: 0,
    animation: "recPulse 1s ease-in-out infinite",
  },
  filledBtn: {
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    padding: "14px 28px",
    background: "rgba(194,24,91,0.9)",
    border: "1.5px solid rgba(194,24,91,0.7)",
    borderRadius: 10,
    color: "#fff", fontSize: 14, fontWeight: 600,
    fontFamily: "'DM Sans', sans-serif",
    letterSpacing: "0.01em",
    cursor: "pointer", transition: "all 0.2s",
    whiteSpace: "nowrap",
    boxShadow: "0 4px 20px rgba(194,24,91,0.3)",
  },
  filledBtnFull: {
    width: "100%",
    marginTop: 8,
  },
  ghostBtn: {
    display: "flex", alignItems: "center", justifyContent: "center",
    width: "100%", marginTop: 10,
    padding: "11px 0",
    background: "transparent",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 10,
    color: "rgba(255,255,255,0.5)",
    fontSize: 13, fontWeight: 500,
    fontFamily: "'DM Sans', sans-serif",
    cursor: "pointer", transition: "all 0.2s",
  },

  // ── Center section
  centerSection: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "0 20px",
    position: "relative",
    zIndex: 1,
  },

  // ── Email screen
  emailCard: {
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 18,
    padding: "36px 32px",
    backdropFilter: "blur(20px)",
    width: "100%",
    maxWidth: 480,
    boxSizing: "border-box",
  },
  emailIcon: {
    width: 56, height: 56, borderRadius: 14,
    background: "rgba(194,24,91,0.15)",
    border: "1px solid rgba(194,24,91,0.25)",
    display: "flex", alignItems: "center", justifyContent: "center",
    marginBottom: 20,
  },
  emailHeading: {
    fontSize: 26, fontWeight: 700, color: "#fff",
    margin: "0 0 8px", letterSpacing: "-0.01em",
  },
  emailSub: {
    fontSize: 15, color: "rgba(255,255,255,0.48)",
    margin: "0 0 24px", lineHeight: 1.55,
  },
  inputWrap: {
    display: "flex", flexDirection: "column", gap: 7,
  },
  inputLabel: {
    fontSize: 13, fontWeight: 600,
    color: "rgba(255,255,255,0.6)",
    letterSpacing: "0.02em",
  },
  underlineInput: {
    width: "100%",
    padding: "13px 16px",
    background: "rgba(255,255,255,0.06)",
    border: "1.5px solid rgba(255,255,255,0.18)",
    borderRadius: 10,
    color: "#fff", fontSize: 16,
    fontFamily: "'DM Sans', sans-serif",
    outline: "none",
    transition: "border-color 0.2s, box-shadow 0.2s, background 0.2s",
    boxSizing: "border-box",
    WebkitAppearance: "none",
    marginBottom: 16,
  },
  errorBox: {
    display: "flex", alignItems: "center", gap: 8,
    padding: "10px 14px",
    background: "rgba(239,68,68,0.1)",
    border: "1px solid rgba(239,68,68,0.25)",
    borderRadius: 9,
    color: "#fca5a5", fontSize: 13,
    lineHeight: 1.4, marginBottom: 8,
  },

  // ── Upload screen
  progressRingWrap: {
    position: "relative",
    display: "flex", alignItems: "center", justifyContent: "center",
    marginBottom: 28,
  },
  progressPct: {
    position: "absolute",
    fontSize: 18, fontWeight: 700,
    fontFamily: "'Space Mono', monospace", color: "#fff",
  },
  uploadTitle: {
    fontSize: 24, fontWeight: 700,
    margin: "0 0 8px", letterSpacing: "-0.01em",
  },
  uploadText: {
    fontSize: 15, color: "rgba(255,255,255,0.48)",
    margin: 0, lineHeight: 1.5,
  },

  // ── Success screen
  checkCircle: {
    width: 88, height: 88, borderRadius: "50%",
    background: "linear-gradient(135deg, rgba(194,24,91,0.45), rgba(123,26,94,0.45))",
    border: "1.5px solid rgba(255,255,255,0.1)",
    display: "flex", alignItems: "center", justifyContent: "center",
    marginBottom: 28, backdropFilter: "blur(8px)",
    boxShadow: "0 8px 36px rgba(194,24,91,0.25)",
  },
  successTitle: {
    fontSize: 32, fontWeight: 700,
    letterSpacing: "-0.01em",
    margin: "0 0 14px",
  },
  successSubtext: {
    fontSize: 15, color: "rgba(255,255,255,0.52)",
    lineHeight: 1.65, textAlign: "center",
    maxWidth: 380, margin: 0,
  },
};

// ─── CSS Injection ────────────────────────────────────────────
if (typeof document !== "undefined") {
  const el = document.createElement("style");
  el.textContent = `
    *, *::before, *::after { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body { margin: 0; overscroll-behavior: none; }

    /* ── Keyframes ──────────────────────────────────────────── */
    @keyframes fadeIn    { from{opacity:0}                             to{opacity:1} }
    @keyframes slideUp   { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
    @keyframes slideDown { from{opacity:0;transform:translateY(-18px)} to{opacity:1;transform:translateY(0)} }
    @keyframes scaleIn   { from{opacity:0;transform:scale(0.88)}       to{opacity:1;transform:scale(1)} }
    @keyframes popIn     { 0%{opacity:0;transform:scale(0.5)} 70%{transform:scale(1.1)} 100%{opacity:1;transform:scale(1)} }
    @keyframes pulse     { 0%,100%{opacity:1} 50%{opacity:0.2} }
    @keyframes recPulse  { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.45;transform:scale(0.82)} }
    @keyframes recordPulse { 0%,100%{box-shadow:0 0 0 0 rgba(229,57,53,0)} 50%{box-shadow:0 0 0 12px rgba(229,57,53,0.15)} }
    @keyframes camGlow { 0%,100%{box-shadow:0 4px 20px rgba(194,24,91,0.35),0 0 0 0 rgba(194,24,91,0)} 50%{box-shadow:0 8px 40px rgba(194,24,91,0.65),0 0 0 14px rgba(194,24,91,0)} }
    @keyframes standbyPulse { 0%,100%{opacity:1} 50%{opacity:0.2} }

    /* ── Animation utility classes ────────────────────────────── */
    .anim-fade-in   { animation: fadeIn    0.5s ease both; }
    .anim-slide-up  { animation: slideUp   0.5s ease both; }
    .anim-slide-down{ animation: slideDown 0.45s ease both; }
    .anim-scale-in  { animation: scaleIn   0.5s ease both; }
    .anim-pop-in    { animation: popIn     0.55s cubic-bezier(.34,1.56,.64,1) both; }

    /* Staggered delays */
    .d1 { animation-delay: 0.05s; }
    .d2 { animation-delay: 0.12s; }
    .d3 { animation-delay: 0.2s;  }
    .d4 { animation-delay: 0.28s; }
    .d5 { animation-delay: 0.38s; }

    /* ── Screen entrance ──────────────────────────────────────── */
    .gradient-screen { animation: fadeIn 0.35s ease both; }

    /* ── Welcome ──────────────────────────────────────────────── */
    .welcome-heading { animation: slideUp 0.6s 0.1s ease both; }

    /* ── Record screen ──────────────────────────────────────── */
    .countdown-num { animation: popIn 0.32s cubic-bezier(.34,1.56,.64,1) both !important; }
    .rec-dot { animation: recPulse 1s ease-in-out infinite !important; }
    .standby-dot { animation: standbyPulse 1.8s ease-in-out infinite !important; }

    /* ── Success ──────────────────────────────────────────────── */
    .success-circle { animation: popIn 0.55s 0.05s cubic-bezier(.34,1.56,.64,1) both !important; }

    /* ── Start button (Welcome CTA) ──────────────────────────── */
    .start-btn {
      padding: 18px 52px !important;
      font-size: 17px !important;
      border-radius: 14px !important;
      letter-spacing: 0.02em !important;
      animation: camGlow 2.2s 1.1s ease-in-out infinite !important;
    }
    @media (max-width: 480px) {
      .start-btn { width: 100% !important; padding: 16px 24px !important; }
    }

    /* ── Button interactions ──────────────────────────────────── */
    .filled-btn:hover  { filter:brightness(1.12); transform:translateY(-2px); box-shadow:0 8px 28px rgba(194,24,91,0.4) !important; }
    .filled-btn:active { transform:scale(0.97) translateY(0) !important; }
    .outline-btn:hover  { background:rgba(255,255,255,0.12) !important; transform:translateY(-1px); }
    .outline-btn:active { transform:scale(0.97) !important; }
    .back-btn:hover  { filter:brightness(1.25); }
    .back-btn:active { transform:scale(0.92); }
    .record-btn:hover { border-color:#fff !important; transform:scale(1.04); }
    .record-btn:active { transform:scale(0.95) !important; }
    .bg-thumb:hover  { transform:scale(1.08); opacity:0.88; }
    .bg-thumb:active { transform:scale(0.94); }
    .hero-rec-btn:after {
      content:''; width:34px; height:34px; borderRadius:50%;
      background:#e53935; display:block; margin:auto;
    }

    /* ── Form focus ──────────────────────────────────────────── */
    .email-input::placeholder { color:rgba(255,255,255,0.22); }
    .email-input:focus { background:rgba(255,255,255,0.09) !important; outline:none; }

    /* ── Scrollbar hide ──────────────────────────────────────── */
    .bg-thumbs::-webkit-scrollbar { display:none; }

    /* ── Responsive ──────────────────────────────────────────── */

    /* Mobile — keep layout comfortable */
    @media (max-width: 480px) {
      .welcome-heading { font-size: 34px !important; }
    }

    /* Tablet — show hero right column */
    @media (min-width: 768px) {
      .hero-right { display: block !important; }
    }

    /* Desktop — larger headings */
    @media (min-width: 1024px) {
      .welcome-heading { font-size: 56px !important; }
    }
  `;
  document.head.appendChild(el);
}
