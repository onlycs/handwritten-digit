const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let lastDrawTime = null;
const AUTO_CLEAR_DELAY = 750;

// Set up canvas
ctx.strokeStyle = "#FFFFFF";
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.fillStyle = "#000000";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

// Touch events for mobile
canvas.addEventListener("touchstart", handleTouch);
canvas.addEventListener("touchmove", handleTouch);
canvas.addEventListener("touchend", stopDrawing);

function startDrawing(e) {
    // Check if we should auto-clear before starting new drawing
    const now = Date.now();
    if (lastDrawTime && now - lastDrawTime >= AUTO_CLEAR_DELAY) {
        clearCanvas();
    }

    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    lastX = e.clientX - rect.left;
    lastY = e.clientY - rect.top;
}

function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    lastX = currentX;
    lastY = currentY;
}

function stopDrawing() {
    isDrawing = false;
    lastDrawTime = Date.now();
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(
        e.type === "touchstart"
            ? "mousedown"
            : e.type === "touchmove"
              ? "mousemove"
              : "mouseup",
        {
            clientX: touch.clientX,
            clientY: touch.clientY,
        },
    );
    canvas.dispatchEvent(mouseEvent);
}

function clearCanvas() {
    lastDrawTime = null;
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result").textContent =
        "Draw a digit to get started!";
    document.getElementById("result").className = "result";
    document.getElementById("confidence").textContent = "";
}

async function predict() {
    // Show loading state
    document.getElementById("result").textContent = "Analyzing...";
    document.getElementById("result").className = "result loading";
    document.getElementById("confidence").textContent = "";

    try {
        // Get canvas data as base64
        const imageData = canvas.toDataURL("image/png");

        // Send to server
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                image: imageData,
            }),
        });

        const result = await response.json();

        if (result.error) {
            document.getElementById("result").textContent = result.error;
            document.getElementById("result").className = "result error";
            document.getElementById("confidence").textContent = "";
        } else {
            // Display the server-processed (centered) image
            if (result.processed_image) {
                const processedImg = new Image();
                processedImg.onload = function () {
                    ctx.fillStyle = "#000000";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.imageSmoothingEnabled = false;
                    ctx.drawImage(processedImg, 0, 0, 280, 280);
                };
                processedImg.src = result.processed_image;
            }

            document.getElementById("result").textContent =
                `Predicted: ${result.digit}`;
            document.getElementById("result").className = "result";
            document.getElementById("confidence").textContent =
                `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
        }
    } catch (error) {
        document.getElementById("result").textContent =
            "Error: Could not connect to server";
        document.getElementById("result").className = "result error";
        document.getElementById("confidence").textContent = "";
    }
}

// Auto-predict after drawing (with debounce)
let autoPredict = true;
let debounceTimer;

canvas.addEventListener("mouseup", () => {
    if (autoPredict) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(predict, 500);
    }
});

canvas.addEventListener("touchend", () => {
    if (autoPredict) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(predict, 500);
    }
});

// Initialize canvas on load
document.addEventListener("DOMContentLoaded", () => {
    clearCanvas();
});
