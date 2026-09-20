/**
 * DIGITIFY — Frontend Canvas Drawing, State Management & Prediction Pipeline
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const canvas = document.getElementById('digit-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const clearBtn = document.getElementById('clear-btn');
  const predictBtn = document.getElementById('predict-btn');
  const predictBtnText = document.getElementById('predict-btn-text');
  const predictSpinner = document.getElementById('predict-spinner');
  const drawHint = document.getElementById('draw-hint');
  const tryItNowBtn = document.getElementById('try-it-now-btn');
  const drawingCard = document.querySelector('.drawing-card');
  const drawingSection = document.getElementById('drawing-section');

  // Result Paper Elements
  const paperContent = document.getElementById('paper-content');
  const predictedDigitEl = document.getElementById('predicted-digit');
  const predictedConfidenceEl = document.getElementById('predicted-confidence');

  // Drawing State
  let isDrawing = false;
  let hasDrawn = false;
  let lastX = 0;
  let lastY = 0;

  // Initialize Canvas Surface with Crisp Retina/HiDPI scaling
  function initCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    // Internal coordinate resolution (280x280)
    canvas.width = 280 * dpr;
    canvas.height = 280 * dpr;
    ctx.scale(dpr, dpr);

    // Initial white background fill
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, 280, 280);

    // Stroke parameters for natural handwriting
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#111827'; // Dark charcoal / black ink
  }

  initCanvas();

  // Resize handler to maintain scale on screen change
  window.addEventListener('resize', () => {
    // Preserve current drawing if resizing
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    initCanvas();
    ctx.putImageData(imgData, 0, 0);
  });

  // Coordinate helper: maps screen/mouse/touch coordinates to 280x280 canvas space
  function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    const scaleX = 280 / rect.width;
    const scaleY = 280 / rect.height;

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }

  // Draw Functions
  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    hideHint();

    const coords = getCoordinates(e);
    lastX = coords.x;
    lastY = coords.y;

    // Draw single dot in case of simple tap/click
    ctx.beginPath();
    ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const coords = getCoordinates(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();

    lastX = coords.x;
    lastY = coords.y;
  }

  function stopDrawing(e) {
    if (!isDrawing) return;
    if (e) e.preventDefault();
    isDrawing = false;
  }

  // Mouse Listeners
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  window.addEventListener('mouseup', stopDrawing);

  // Touch Listeners (Mobile / Tablet / Stylus)
  canvas.addEventListener('touchstart', startDrawing, { passive: false });
  canvas.addEventListener('touchmove', draw, { passive: false });
  window.addEventListener('touchend', stopDrawing, { passive: false });
  window.addEventListener('touchcancel', stopDrawing, { passive: false });

  // Show Toast Hint
  let hintTimeout;
  function showHint(msg = 'Draw a digit first.') {
    drawHint.textContent = msg;
    drawHint.classList.add('visible');
    clearTimeout(hintTimeout);
    hintTimeout = setTimeout(() => {
      drawHint.classList.remove('visible');
    }, 2400);
  }

  function hideHint() {
    drawHint.classList.remove('visible');
  }

  // Check if Canvas has Actual Drawing Content
  function isCanvasBlank() {
    const pixelData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let darkPixels = 0;
    // Inspect red channel (grayscale)
    for (let i = 0; i < pixelData.length; i += 4) {
      if (pixelData[i] < 200) { // If pixel is significantly darker than pure white (255)
        darkPixels++;
        if (darkPixels > 60) return false; // Found sufficient stroke content
      }
    }
    return true;
  }

  // Clear Canvas and Reset Result State
  function clearAll() {
    initCanvas();
    hasDrawn = false;
    hideHint();

    // Reset prediction paper to completely empty state
    paperContent.classList.remove('forming');
    paperContent.classList.add('empty');
    predictedDigitEl.textContent = '-';
    predictedConfidenceEl.textContent = '-%';
  }

  clearBtn.addEventListener('click', clearAll);

  // "Try It Now" Button: Smoothly scrolls and focuses the drawing card
  tryItNowBtn.addEventListener('click', () => {
    drawingSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    drawingCard.classList.remove('highlight-focus');
    // Trigger reflow to restart pulse animation
    void drawingCard.offsetWidth;
    drawingCard.classList.add('highlight-focus');
  });

  // Predict Action: Prepares Canvas Data URL, POSTs to /predict, and Triggers Forming Animation
  async function handlePredict() {
    if (!hasDrawn || isCanvasBlank()) {
      showHint('Draw a digit first.');
      return;
    }

    // Set Loading State
    predictBtn.disabled = true;
    predictBtnText.textContent = 'Predicting...';
    predictSpinner.classList.remove('hidden');

    try {
      // Export canvas drawing as standard PNG Data URL
      const dataURL = canvas.toDataURL('image/png');

      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL })
      });

      const result = await response.json();

      if (!response.ok) {
        showHint(result.error || 'Prediction failed. Try again.');
        return;
      }

      // Display prediction with back-to-front physical reveal animation
      displayPrediction(result.prediction, result.confidence);

    } catch (err) {
      console.error('Prediction network error:', err);
      showHint('Backend unavailable. Please try again.');
    } finally {
      // Restore button state
      predictBtn.disabled = false;
      predictBtnText.textContent = 'Predict';
      predictSpinner.classList.add('hidden');
    }
  }

  predictBtn.addEventListener('click', handlePredict);

  // Animate and Populate Prediction on Torn Paper
  function displayPrediction(digit, confidence) {
    // Populate values
    predictedDigitEl.textContent = digit;
    predictedConfidenceEl.textContent = `${(confidence * 100).toFixed(1)}%`;

    // Trigger deliberate back-to-front forming sequence
    paperContent.classList.remove('empty', 'forming');
    void paperContent.offsetWidth; // Force CSS reflow
    paperContent.classList.add('forming');
  }
});

