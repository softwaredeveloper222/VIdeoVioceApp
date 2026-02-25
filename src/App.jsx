import { useState, useRef, useEffect, useCallback } from "react";

const GRADIENT_BG = "linear-gradient(155deg, #0a1628 0%, #162052 20%, #3b1760 45%, #7b1a5e 70%, #c2185b 95%)";

// ─── MediaPipe Configuration ──────────────────────────────────
const MEDIAPIPE_WASM_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm";
const SELFIE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";

// Segmentation runs on a downscaled canvas for performance.
// 1280×720 → 640×360 = 4× fewer pixels for ML inference.
// GPU bilinear interpolation upscales the mask in the shader.
const SEG_WIDTH = 640;
const SEG_HEIGHT = 360;

const BACKGROUNDS = [
  { id: "none", label: "None", type: "none", preview: "#1a1a2e" },
  { id: "living-room", label: "Living Room", type: "image", src: "/backgrounds/living-room.jpg", preview: "linear-gradient(135deg, #c8956a, #f0ebe0)" },
  { id: "home-office", label: "Home Office", type: "image", src: "/backgrounds/home-office.jpg", preview: "linear-gradient(135deg, #7a9ab0, #e8ecf0)" },
  { id: "library", label: "Library", type: "image", src: "/backgrounds/library.jpg", preview: "linear-gradient(135deg, #7a5c3c, #c8a050)" },
  { id: "cafe", label: "Cafe", type: "image", src: "/backgrounds/cafe.jpg", preview: "linear-gradient(135deg, #b07850, #e8d090)" },
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
            runningMode: "VIDEO",
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          });
        } catch {
          segmenter = await ImageSegmenter.createFromOptions(vision, {
            baseOptions: { modelAssetPath: SELFIE_MODEL_URL, delegate: "CPU" },
            runningMode: "VIDEO",
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
uniform sampler2D u_mask;     // temporally blended mask — stable background/foreground
uniform sampler2D u_rawMask;  // current-frame raw mask — zero temporal lag
uniform sampler2D u_bg;
uniform int u_mode;
uniform vec2 u_texelSize;
uniform vec2 u_maskTexelSize;
in vec2 v_uv;
out vec4 fragColor;
void main() {
  vec4 vid = texture(u_video, v_uv);
  if (u_mode == 0) { fragColor = vid; return; }

  // ── 3×3 Gaussian + dilation on BLENDED mask ───────────────────────────────
  // 3×3 (vs old 5×5) = thinner feather zone, reducing visible edge thickness.
  // Each mask pixel is ~5 video pixels at 256→1280 scale; 3×3 covers ~15 video
  // pixels across the boundary (5×5 was ~25), giving a noticeably slimmer edge.
  const float gw3[3] = float[](0.25, 0.5, 0.25);
  float mBlur = 0.0;
  float mDilate = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      float s = texture(u_mask, v_uv + vec2(float(dx), float(dy)) * u_maskTexelSize).r;
      mBlur += s * gw3[dx+1] * gw3[dy+1];
      mDilate = max(mDilate, s);
    }
  }
  float mFeat = max(mBlur, mDilate * 0.1);

  // ── Guided filter 3×3 using RAW mask for P ────────────────────────────────
  // KEY FIX: P is sampled from u_rawMask (current frame, no temporal blending).
  // The linear model a·I + b is therefore fitted to where the boundary IS NOW,
  // not where it was in the blended history. The resulting mGuided always snaps
  // composite edges to the current video frame's colour boundary — zero lag.
  float guide = dot(vid.rgb, vec3(0.299, 0.587, 0.114));
  float sI = 0.0, sP = 0.0, sIP = 0.0, sII = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2 uv2 = v_uv + vec2(float(dx), float(dy)) * u_maskTexelSize;
      float I = dot(texture(u_video, uv2).rgb, vec3(0.299, 0.587, 0.114));
      float P = texture(u_rawMask, uv2).r;
      sI += I; sP += P; sIP += I * P; sII += I * I;
    }
  }
  float n = 9.0;
  float mI = sI / n, mP = sP / n;
  float a = (sIP / n - mI * mP) / (sII / n - mI * mI + 0.02);
  float b = mP - a * mI;
  float mGuided = clamp(a * guide + b, 0.0, 1.0);

  // ── Composite ─────────────────────────────────────────────────────────────
  // maskUncertainty peaks at 1 where mFeat ≈ 0.5 (the silhouette edge) and
  // falls to 0 in confident foreground/background. At the edge: push guided
  // weight to 1.0 so the boundary is fully current-frame-anchored (no lag).
  // Away from the edge: 70 % guided + 30 % Gaussian gives soft feathering.
  float maskUncertainty = 1.0 - abs(mFeat * 2.0 - 1.0);
  float m = mix(mFeat, mGuided, 0.7 + maskUncertainty * 0.3);

  // Tight smoothstep: 0.40–0.60 is a 0.20-unit band (old 0.15–0.85 = 0.70).
  // Reduces visible edge thickness to ~3–4 video pixels on typical hardware.
  m = smoothstep(0.40, 0.60, m);

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
  // LINEAR filtering on mask gives bilinear interpolation; the joint bilateral
  // shader filter prevents this from creating halo at color-discontinuity edges.
  const hasFloatLinear = gl.getExtension("OES_texture_float_linear");
  const maskFilter = hasFloatLinear ? gl.LINEAR : gl.NEAREST;

  // Compile shaders & link program
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

  // Fullscreen quad: positions + UVs (TRIANGLE_STRIP)
  // UV Y is flipped so video isn't upside-down
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

  // Textures
  const videoTex = createGLTexture(gl, gl.LINEAR);
  const maskTex = createGLTexture(gl, maskFilter);      // unit 1 — blended mask
  const rawMaskTex = createGLTexture(gl, maskFilter);   // unit 3 — current-frame raw mask
  const bgTex = createGLTexture(gl, gl.LINEAR);
  // Dark fallback so unloaded backgrounds don't show garbage
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([26, 26, 46, 255]));

  // Cache uniform locations
  gl.useProgram(program);
  const uniforms = {
    u_video: gl.getUniformLocation(program, "u_video"),
    u_mask: gl.getUniformLocation(program, "u_mask"),
    u_rawMask: gl.getUniformLocation(program, "u_rawMask"),
    u_bg: gl.getUniformLocation(program, "u_bg"),
    u_mode: gl.getUniformLocation(program, "u_mode"),
    u_texelSize: gl.getUniformLocation(program, "u_texelSize"),
  };
  // Bind texture units once (they never change)
  gl.uniform1i(uniforms.u_video, 0);
  gl.uniform1i(uniforms.u_mask, 1);
  gl.uniform1i(uniforms.u_bg, 2);
  gl.uniform1i(uniforms.u_rawMask, 3);

  // Set mask texel size once — mask is always SEG_WIDTH×SEG_HEIGHT regardless
  // of the video resolution, so this is a true constant for the shader.
  const u_maskTexelSizeLoc = gl.getUniformLocation(program, "u_maskTexelSize");
  gl.uniform2f(u_maskTexelSizeLoc, 1.0 / SEG_WIDTH, 1.0 / SEG_HEIGHT);

  return { program, vao, buf, textures: { video: videoTex, mask: maskTex, rawMask: rawMaskTex, bg: bgTex }, uniforms };
}

// ─── WebGL background compositing with ML segmentation ────────
function useBackgroundEffect(videoRef, canvasRef, selectedBg, segmenterRef, segmenterReady, uploadedImage, bgImagesRef) {
  // Refs to track changing values without re-running the effect
  const selectedBgRef = useRef(selectedBg);
  const segmenterReadyRef = useRef(segmenterReady);
  const uploadedImageRef = useRef(uploadedImage);
  selectedBgRef.current = selectedBg;
  segmenterReadyRef.current = segmenterReady;
  uploadedImageRef.current = uploadedImage;

  const rendererRef = useRef(null);
  const blurCanvasRef = useRef(null);
  const blurCtxRef = useRef(null);
  const lastTimeRef = useRef(0);
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
  const skipIntervalRef = useRef(2);

  // Initialize WebGL once, draw loop reads from refs
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false, alpha: false });
    if (!gl) { console.error("WebGL2 not supported"); return; }
    rendererRef.current = initWebGL(gl);

    blurCanvasRef.current = document.createElement("canvas");
    blurCtxRef.current = blurCanvasRef.current.getContext("2d");

    // Small canvas for downscaled segmentation input
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

      // Resize only when dimensions change; pre-allocate video texture so
      // per-frame uploads can use texSubImage2D (no GPU reallocation each frame).
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, r.textures.video);
      if (lastDimsRef.current.w !== w || lastDimsRef.current.h !== h) {
        canvas.width = w;
        canvas.height = h;
        gl.viewport(0, 0, w, h);
        gl.useProgram(r.program);
        gl.uniform2f(r.uniforms.u_texelSize, 1.0 / w, 1.0 / h);
        // Allocate storage once at the new size; subsequent frames use texSubImage2D.
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        lastDimsRef.current = { w, h };
      }

      // Upload video frame — texSubImage2D reuses existing GPU allocation (no realloc).
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, video);

      // "none" mode or segmenter not ready — passthrough video
      if (curBg === "none" || !curReady || !segmenterRef.current) {
        gl.useProgram(r.program);
        gl.uniform1i(r.uniforms.u_mode, 0);
        gl.bindVertexArray(r.vao);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        animFrameRef.current = requestAnimationFrame(draw);
        return;
      }

      // Run ML segmentation on a downscaled canvas at adaptive intervals.
      // 1280×720 → 640×360 = 4× fewer pixels. Skip interval adapts to device (1–4 frames).
      frameCountRef.current++;
      const shouldSegment = frameCountRef.current % skipIntervalRef.current === 0;

      if (shouldSegment) {
        const segStart = performance.now();
        const timestamp = segStart > lastTimeRef.current ? segStart : lastTimeRef.current + 1;
        lastTimeRef.current = timestamp;

        // Draw video to small canvas for fast segmentation
        segCtxRef.current.drawImage(video, 0, 0, SEG_WIDTH, SEG_HEIGHT);

        let mask = null;
        try {
          const result = segmenterRef.current.segmentForVideo(segCanvasRef.current, timestamp);
          if (result.confidenceMasks?.length > 0) {
            mask = result.confidenceMasks[0].getAsFloat32Array();
          }
        } catch { /* fall through */ }

        if (mask) {
          // ── Upload raw mask to texture unit 3 FIRST (before any blending) ──
          // The shader's guided filter reads u_rawMask for its P samples so its
          // linear model is fitted to the boundary's current position — zero lag.
          gl.activeTexture(gl.TEXTURE3);
          gl.bindTexture(gl.TEXTURE_2D, r.textures.rawMask);
          if (maskAllocRef.current.w !== SEG_WIDTH || maskAllocRef.current.h !== SEG_HEIGHT) {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, SEG_WIDTH, SEG_HEIGHT, 0, gl.RED, gl.FLOAT, mask);
          } else {
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SEG_WIDTH, SEG_HEIGHT, gl.RED, gl.FLOAT, mask);
          }

          // ── Per-pixel boundary-aware temporal blend → texture unit 1 ───────
          // globalAlpha: low when static (stable BG/FG), high during motion.
          // localAlpha: at boundary pixels (mask ≈ 0.5) always → 0.97 so the
          // silhouette edge tracks without lag in the blended mask too.
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
              const localAlpha = globalAlpha + uncertainty * (0.97 - globalAlpha);
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

        // Adaptive load management — keep skip interval at 1 for most devices
        const segTime = performance.now() - segStart;
        const avg = avgFrameTimeRef.current = avgFrameTimeRef.current * 0.9 + segTime * 0.1;
        if (avg > 8 && skipIntervalRef.current < 4) skipIntervalRef.current++;
        else if (avg < 3 && skipIntervalRef.current > 1) skipIntervalRef.current--;
      }

      // No valid mask yet — passthrough until first segmentation completes
      if (!hasMaskRef.current) {
        gl.useProgram(r.program);
        gl.uniform1i(r.uniforms.u_mode, 0);
        gl.bindVertexArray(r.vao);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        animFrameRef.current = requestAnimationFrame(draw);
        return;
      }

      // Upload background texture
      const bg = BACKGROUNDS.find((b) => b.id === curBg);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, r.textures.bg);

      if (bg?.type === "blur") {
        // Update blur background every frame at half resolution for smooth motion.
        // Half-res (w/2 × h/2) gives 4× fewer pixels than full-res while retaining
        // much better quality than quarter-res; the CSS blur radius is halved
        // proportionally so the effective visual blur stays at bg.blurPx video pixels.
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

      // Draw composite — single GPU draw call
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

  // Preload preset background images
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

        {!segmenterReady && !segmenterError && selectedBg !== "none" && phase !== "preview" && (
          <div style={styles.segmenterLoadingOverlay}>
            <span style={styles.segmenterLoadingDot} />
            <span style={styles.segmenterLoadingLabel}>Loading AI background...</span>
          </div>
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
            <p style={styles.bgTitle}>
              Choose your background
              {!segmenterReady && !segmenterError && (
                <span style={styles.segmenterLoadingText}> (AI loading...)</span>
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
                  {bg.type === "upload" && !uploadedImage && (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                      stroke="rgba(255,255,255,0.6)" strokeWidth="2"
                      strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                  )}
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
    position: "absolute",
    inset: 0,
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

  // ─── Segmenter loading
  segmenterLoadingOverlay: {
    position: "absolute",
    top: 16,
    right: 16,
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "6px 14px",
    background: "rgba(0,0,0,0.5)",
    borderRadius: 20,
    backdropFilter: "blur(8px)",
    zIndex: 5,
  },
  segmenterLoadingDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#ffa726",
    animation: "pulse 1.5s infinite",
  },
  segmenterLoadingLabel: {
    fontSize: 11,
    fontWeight: 500,
    color: "rgba(255,255,255,0.7)",
  },
  segmenterLoadingText: {
    fontSize: 11,
    fontWeight: 400,
    color: "rgba(255,255,255,0.35)",
    fontStyle: "italic",
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

  // ─── Bottom panel (camera screen) — floats over the camera feed
  bottomPanel: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    background: "linear-gradient(to top, rgba(0,0,0,0.45) 60%, transparent)",
    padding: "32px 24px 36px",
    zIndex: 10,
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
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundSize: "cover",
    backgroundPosition: "center",
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
