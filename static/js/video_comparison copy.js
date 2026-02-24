// Written by Dor Verbin, October 2021
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    var position = 0.5;
    var vidWidth = vid.videoWidth/2;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    
    if (vid.readyState > 3) {
        vid.play();

        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - bcr.x) / bcr.width);
        }

        videoMerge.addEventListener("mousemove",  trackLocation, false); 
        videoMerge.addEventListener("touchstart", trackLocationTouch, false);
        videoMerge.addEventListener("touchmove",  trackLocationTouch, false);


        function drawLoop() {
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);
            mergeContext.drawImage(vid, colStart+vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
            requestAnimationFrame(drawLoop);

            
            var arrowLength = 0.09 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 10;
            var arrowWidth = 0.007 * vidHeight;
            var currX = vidWidth * position;

            // Draw circle
            mergeContext.arc(currX, arrowPosY, arrowLength*0.7, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#FFD79340";
            mergeContext.fill()
            //mergeContext.strokeStyle = "#444444";
            //mergeContext.stroke()
            
            // Draw border
            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth*position, 0);
            mergeContext.lineTo(vidWidth*position, vidHeight);
            mergeContext.closePath()
            mergeContext.strokeStyle = "#AAAAAA";
            mergeContext.lineWidth = 5;            
            mergeContext.stroke();

            // Draw arrow
            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth/2);
            
            // Move right until meeting arrow head
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowWidth/2);
            
            // Draw right arrow head
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowheadWidth/2);
            mergeContext.lineTo(currX + arrowLength/2, arrowPosY);
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowheadWidth/2);
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowWidth/2);

            // Go back to the left until meeting left arrow head
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowWidth/2);
            
            // Draw left arrow head
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowheadWidth/2);
            mergeContext.lineTo(currX - arrowLength/2, arrowPosY);
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY  - arrowheadWidth/2);
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY);
            
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY - arrowWidth/2);
            mergeContext.lineTo(currX, arrowPosY - arrowWidth/2);

            mergeContext.closePath();

            mergeContext.fillStyle = "#AAAAAA";
            mergeContext.fill();

            
            
        }
        requestAnimationFrame(drawLoop);
    } 
}

Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};
    
    
function resizeAndPlay(element)
{
  var cv = document.getElementById(element.id + "Merge");
  cv.width = element.videoWidth/2;
  cv.height = element.videoHeight;
  element.play();
  element.style.height = "0px";  // Hide video without stopping it
    
  playVids(element.id);
}


function playVids2(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    var startPosition = 1 / 3; 
    var endPosition = 2 / 3;   
    var vidWidth = vid.videoWidth / 3;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");
    var isDraggingStart = false;  
    var isDraggingEnd = false;   

    var arrowLength = 0.09 * vidHeight;
    var arrowheadWidth = 0.025 * vidHeight;
    var arrowheadLength = 0.04 * vidHeight;
    var arrowWidth = 0.007 * vidHeight;
    var circleRadius = arrowLength * 0.7;
    var arrowPosY = vidHeight / 10;

    if (vid.readyState > 3) {
        vid.play();

        function trackStartLocation(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.pageX - bcr.x) / bcr.width);
            startPosition = newPos;
            
            if (startPosition > 0.99) {
                startPosition = 0.99;
            } else if (startPosition < 0.01) {
                startPosition = 0.01;
            }
            if (startPosition > endPosition) {
                endPosition = startPosition; 
            }
        }

        function trackEndLocation(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.pageX - bcr.x) / bcr.width);
            endPosition = newPos;
            
            if (endPosition > 0.99) {
                endPosition = 0.99;
            } else if (endPosition < 0.01) {
                endPosition = 0.01;
            }
            if (endPosition < startPosition) {
                startPosition = endPosition;  
            }
        }

        function trackStartLocationTouch(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.touches[0].pageX - bcr.x) / bcr.width);
            startPosition = newPos;
            
            if (startPosition > 0.99) {
                startPosition = 0.99;
            } else if (startPosition < 0.01) {
                startPosition = 0.01;
            }
            if (startPosition > endPosition) {
                endPosition = startPosition; 
            }
        }

        function trackEndLocationTouch(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.touches[0].pageX - bcr.x) / bcr.width);
            endPosition = newPos;
            
            if (endPosition > 0.99) {
                endPosition = 0.99;
            } else if (endPosition < 0.01) {
                endPosition = 0.01;
            }
            if (endPosition < startPosition) {
                startPosition = endPosition;  
            }
        }

        videoMerge.addEventListener("mousedown", function(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var clickPos = (e.pageX - bcr.x) / bcr.width;
            var adjustedPageY = e.pageY - window.scrollY;
            var clickPosY = (adjustedPageY - bcr.y) / bcr.height;

            if ((Math.abs(clickPos - startPosition) <= 0.05) && (Math.abs(clickPosY - 0.1) <= 0.05)) {
                isDraggingStart = true;
            }
            else if ((Math.abs(clickPos - endPosition) <= 0.05) && (Math.abs(clickPosY - 0.9) <= 0.05)) {
                isDraggingEnd = true;
            }
            e.preventDefault();
            document.body.style.userSelect = "none";
        }, false);

        videoMerge.addEventListener("mousemove", function(e) {
            if (isDraggingStart) {
                trackStartLocation(e); 
            } else if (isDraggingEnd) {
                trackEndLocation(e); 
            }
        }, false);

        document.addEventListener("mouseup", function() {
            isDraggingStart = false;
            isDraggingEnd = false;
            document.body.style.userSelect = "";
        }, false);

        videoMerge.addEventListener("touchstart", function(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var clickPos = (e.touches[0].pageX - bcr.x) / bcr.width;
            var adjustedPageY = e.touches[0].pageY - window.scrollY;
            var clickPosY = (adjustedPageY - bcr.y) / bcr.height;

            if ((Math.abs(clickPos - startPosition) <= 0.05) && (Math.abs(clickPosY - 0.1) <= 0.05)) {
                isDraggingStart = true;
            }
            else if ((Math.abs(clickPos - endPosition) <= 0.05) && (Math.abs(clickPosY - 0.9) <= 0.05)) {
                isDraggingEnd = true;
            }
            e.preventDefault();
            document.body.style.userSelect = "none";
        }, false);

        videoMerge.addEventListener("touchmove", function(e) {
            if (isDraggingStart) {
                trackStartLocationTouch(e); 
            } else if (isDraggingEnd) {
                trackEndLocationTouch(e);
            }
        }, false);

        document.addEventListener("touchend", function() {
            isDraggingStart = false;
            isDraggingEnd = false;
            document.body.style.userSelect = "";
        }, false);
        
        function drawLoop() {
            mergeContext.clearRect(0, 0, vidWidth, vidHeight);

            var colStart1 = (vidWidth * startPosition).clamp(0.0, vidWidth);
            var colStart2 = (vidWidth * endPosition).clamp(0.0, vidWidth);

            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight); 
            mergeContext.drawImage(vid, colStart1 + vidWidth, 0, colStart2 - colStart1, vidHeight, colStart1, 0, colStart2 - colStart1, vidHeight);
            mergeContext.drawImage(vid, colStart2 + vidWidth*2, 0, vidWidth - colStart2, vidHeight, colStart2, 0, vidWidth - colStart2, vidHeight);

            mergeContext.beginPath();
            mergeContext.moveTo(colStart1, 0);  
            mergeContext.lineTo(colStart1, vidHeight);
            mergeContext.moveTo(colStart2, 0); 
            mergeContext.lineTo(colStart2, vidHeight);
            mergeContext.strokeStyle = "#AAAAAA"; 
            mergeContext.lineWidth = 5;
            mergeContext.stroke();

            var currX = vidWidth * startPosition;

            mergeContext.beginPath();
            mergeContext.arc(currX, arrowPosY, circleRadius, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#FFD79340"; 
            mergeContext.fill();
            mergeContext.closePath();

            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth / 3);
            mergeContext.lineTo(currX + arrowLength / 3 - arrowheadLength / 3, arrowPosY - arrowWidth / 3);
            mergeContext.lineTo(currX + arrowLength / 3 - arrowheadLength / 3, arrowPosY - arrowheadWidth / 3);
            mergeContext.lineTo(currX + arrowLength / 3, arrowPosY);
            mergeContext.lineTo(currX + arrowLength / 3 - arrowheadLength / 3, arrowPosY + arrowheadWidth / 3);
            mergeContext.lineTo(currX + arrowLength / 3 - arrowheadLength / 3, arrowPosY + arrowWidth / 3);
            mergeContext.lineTo(currX - arrowLength / 3 + arrowheadLength / 3, arrowPosY + arrowWidth / 3);
            mergeContext.lineTo(currX - arrowLength / 3 + arrowheadLength / 3, arrowPosY + arrowheadWidth / 3);
            mergeContext.lineTo(currX - arrowLength / 3, arrowPosY);
            mergeContext.lineTo(currX - arrowLength / 3 + arrowheadLength / 3, arrowPosY - arrowheadWidth / 3);
            mergeContext.lineTo(currX - arrowLength / 3 + arrowheadLength / 3, arrowPosY);
            mergeContext.lineTo(currX - arrowLength / 3 + arrowheadLength / 3, arrowPosY - arrowWidth / 3);
            mergeContext.lineTo(currX, arrowPosY - arrowWidth / 3);
            mergeContext.closePath();
            mergeContext.fillStyle = "#AAAAAA";
            mergeContext.fill();

            var currX2 = vidWidth * endPosition;
            var arrowPosY2 = vidHeight - (vidHeight / 10); 

            mergeContext.beginPath();
            mergeContext.arc(currX2, arrowPosY2, circleRadius, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#FFD79340"; 
            mergeContext.fill();
            mergeContext.closePath();

            mergeContext.beginPath();
            mergeContext.moveTo(currX2, arrowPosY2 - arrowWidth / 3);
            mergeContext.lineTo(currX2 + arrowLength / 3 - arrowheadLength / 3, arrowPosY2 - arrowWidth / 3);
            mergeContext.lineTo(currX2 + arrowLength / 3 - arrowheadLength / 3, arrowPosY2 - arrowheadWidth / 3);
            mergeContext.lineTo(currX2 + arrowLength / 3, arrowPosY2);
            mergeContext.lineTo(currX2 + arrowLength / 3 - arrowheadLength / 3, arrowPosY2 + arrowheadWidth / 3);
            mergeContext.lineTo(currX2 + arrowLength / 3 - arrowheadLength / 3, arrowPosY2 + arrowWidth / 3);
            mergeContext.lineTo(currX2 - arrowLength / 3 + arrowheadLength / 3, arrowPosY2 + arrowWidth / 3);
            mergeContext.lineTo(currX2 - arrowLength / 3 + arrowheadLength / 3, arrowPosY2 + arrowheadWidth / 3);
            mergeContext.lineTo(currX2 - arrowLength / 3, arrowPosY2);
            mergeContext.lineTo(currX2 - arrowLength / 3 + arrowheadLength / 3, arrowPosY2 - arrowheadWidth / 3);
            mergeContext.lineTo(currX2 - arrowLength / 3 + arrowheadLength / 3, arrowPosY2);
            mergeContext.lineTo(currX2 - arrowLength / 3 + arrowheadLength / 3, arrowPosY2 - arrowWidth / 3);
            mergeContext.lineTo(currX2, arrowPosY2 - arrowWidth / 3);
            mergeContext.closePath();
            mergeContext.fillStyle = "#AAAAAA";
            mergeContext.fill();

            requestAnimationFrame(drawLoop);
        }

        requestAnimationFrame(drawLoop);
    }
}

Number.prototype.clamp = function(min, max) {
    return Math.min(Math.max(this, min), max);
};


function resizeAndPlay2(element) {
    var cv = document.getElementById(element.id + "Merge");
    cv.width = element.videoWidth / 3;
    cv.height = element.videoHeight;
    element.play();
    element.style.height = "0px";  // Hide video without stopping it
    playVids2(element.id);
}