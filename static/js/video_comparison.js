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
    cv.width = element.videoWidth / 5;
    cv.height = element.videoHeight;
    element.play();
    element.style.height = "0px";  // Hide video without stopping it
    playVids2(element.id);
}

function playVids5(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    var Pos1 = 1 / 5; 
    var Pos2 = 2 / 5; 
    var Pos3 = 3 / 5; 
    var Pos4 = 4 / 5;   
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

        function track1Location(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.pageX - bcr.x) / bcr.width);
            Pos1 = newPos;
            
            if (Pos1 > 0.99) {
                Pos1 = 0.99;
            } else if (Pos1 < 0.01) {
                Pos1 = 0.01;
            }
            if (Pos1 > Pos2) {
                Pos2 = Pos1; 
            }
        }
        function track2Location(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.pageX - bcr.x) / bcr.width);
            Pos2 = newPos;
            
            if (Pos2 > 0.99) {
                Pos2 = 0.99;
            } else if (Pos2 < 0.01) {
                Pos2 = 0.01;
            }
            if (Pos2 > Pos3) {
                Pos3 = Pos2; 
            }
        }
        function track3Location(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.pageX - bcr.x) / bcr.width);
            Pos3 = newPos;
            
            if (Pos3 > 0.99) {
                Pos3 = 0.99;
            } else if (Pos3 < 0.01) {
                Pos3 = 0.01;
            }
            if (Pos3 > Pos4) {
                Pos4 = Pos3; 
            }
        }
        function track4Location(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.pageX - bcr.x) / bcr.width);
            Pos4 = newPos;
            
            if (Pos4 > 0.99) {
                Pos4 = 0.99;
            } else if (Pos4 < 0.01) {
                Pos4 = 0.01;
            }
            if (Pos4 < Pos3) {
                Pos3 = Pos4; 
            }
        }

        function track1LocationTouch(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.touches[0].pageX - bcr.x) / bcr.width);
            Pos1 = newPos;
            
            if (Pos1 > 0.99) {
                Pos1 = 0.99;
            } else if (Pos1 < 0.01) {
                Pos1 = 0.01;
            }
            if (Pos1 > Pos2) {
                Pos2 = Pos1; 
            }
        }
        function track2LocationTouch(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.touches[0].pageX - bcr.x) / bcr.width);
            Pos2 = newPos;
            
            if (Pos2 > 0.99) {
                Pos2 = 0.99;
            } else if (Pos2 < 0.01) {
                Pos2 = 0.01;
            }
            if (Pos2 > Pos3) {
                Pos3 = Pos2; 
            }
        }
        function track3LocationTouch(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.touches[0].pageX - bcr.x) / bcr.width);
            Pos3 = newPos;
            
            if (Pos3 > 0.99) {
                Pos3 = 0.99;
            } else if (Pos3 < 0.01) {
                Pos3 = 0.01;
            }
            if (Pos3 > Pos4) {
                Pos4 = Pos3; 
            }
        }
        function track4LocationTouch(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var newPos = ((e.touches[0].pageX - bcr.x) / bcr.width);
            Pos4 = newPos;
            
            if (Pos4 > 0.99) {
                Pos4 = 0.99;
            } else if (Pos4 < 0.01) {
                Pos4 = 0.01;
            }
            if (Pos4 < Pos3) {
                Pos3 = Pos4; 
            }
        }



        videoMerge.addEventListener("mousedown", function(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var clickPos = (e.pageX - bcr.x) / bcr.width;
            var adjustedPageY = e.pageY - window.scrollY;
            var clickPosY = (adjustedPageY - bcr.y) / bcr.height;
        
            // 예시: 각 handle의 수평 기준은 Pos1, Pos2, Pos3, Pos4이고,
            // 각 handle의 세로 기준은 각각 0.1, 0.3, 0.7, 0.9 (0~1 범위)로 가정합니다.
            if ((Math.abs(clickPos - Pos1) <= 0.05) && (Math.abs(clickPosY - 0.1) <= 0.05)) {
                draggingHandle = 1; // 첫번째 handle
            } else if ((Math.abs(clickPos - Pos2) <= 0.05) && (Math.abs(clickPosY - 0.3) <= 0.05)) {
                draggingHandle = 2; // 두번째 handle
            } else if ((Math.abs(clickPos - Pos3) <= 0.05) && (Math.abs(clickPosY - 0.7) <= 0.05)) {
                draggingHandle = 3; // 세번째 handle
            } else if ((Math.abs(clickPos - Pos4) <= 0.05) && (Math.abs(clickPosY - 0.9) <= 0.05)) {
                draggingHandle = 4; // 네번째 handle
            }
            e.preventDefault();
            document.body.style.userSelect = "none";
        }, false);

        videoMerge.addEventListener("mousemove", function(e) {
            if (draggingHandle === 1) {
              track1Location(e);
            } else if (draggingHandle === 2) {
              track2Location(e);
            } else if (draggingHandle === 3) {
              track3Location(e);
            } else if (draggingHandle === 4) {
              track4Location(e);
            }
        }, false);
          

        document.addEventListener("mouseup", function() {
            draggingHandle = null;
            document.body.style.userSelect = "";
        }, false);

        videoMerge.addEventListener("touchstart", function(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var clickPos = (e.touches[0].pageX - bcr.x) / bcr.width;
            var adjustedPageY = e.touches[0].pageY - window.scrollY;
            var clickPosY = (adjustedPageY - bcr.y) / bcr.height;
            
            if ((Math.abs(clickPos - Pos1) <= 0.05) && (Math.abs(clickPosY - 0.1) <= 0.05)) {
                draggingHandle = 1;
            } else if ((Math.abs(clickPos - Pos2) <= 0.05) && (Math.abs(clickPosY - 0.3) <= 0.05)) {
                draggingHandle = 2;
            } else if ((Math.abs(clickPos - Pos3) <= 0.05) && (Math.abs(clickPosY - 0.7) <= 0.05)) {
                draggingHandle = 3;
            } else if ((Math.abs(clickPos - Pos4) <= 0.05) && (Math.abs(clickPosY - 0.9) <= 0.05)) {
                draggingHandle = 4;
            }
            e.preventDefault();
            document.body.style.userSelect = "none";
        }, false);
        
        // 터치 이벤트: handle 드래그 (현재 draggingHandle 값에 따라 해당 handle 업데이트)
        videoMerge.addEventListener("touchmove", function(e) {
            if (draggingHandle === 1) {
                track1LocationTouch(e);
            } else if (draggingHandle === 2) {
                track2LocationTouch(e);
            } else if (draggingHandle === 3) {
                track3LocationTouch(e);
            } else if (draggingHandle === 4) {
                track4LocationTouch(e);
            }
            e.preventDefault();
        }, false);
        
        // 터치 종료 시 드래그 상태 해제
        document.addEventListener("touchend", function() {
            draggingHandle = null;
            document.body.style.userSelect = "";
        }, false);

        
        function drawLoop() {
            // 캔버스 전체 지우기
            mergeContext.clearRect(0, 0, cv.width, cv.height);
          
            // 원본 비디오의 5등분 세그먼트 너비
            var segWidth = vidWidth / 5;
            var segHeight = vidHeight;
          
            // 각 경계의 픽셀 위치 (0~segWidth 범위에서)
            var b1 = (segWidth * Pos1).clamp(0, segWidth);
            var b2 = (segWidth * Pos2).clamp(0, segWidth);
            var b3 = (segWidth * Pos3).clamp(0, segWidth);
            var b4 = (segWidth * Pos4).clamp(0, segWidth);
          
            // 각 구간을 캔버스에 그리기  
            // 구간 1: 원본 비디오의 0 ~ segWidth 구간의 좌측 b1만큼 보여짐
            mergeContext.drawImage(vid, 0, 0, segWidth, segHeight, 0, 0, b1, segHeight);
            // 구간 2: 원본 비디오의 segWidth ~ 2*segWidth 구간의 b2-b1만큼 보여짐
            mergeContext.drawImage(vid, segWidth, 0, segWidth, segHeight, b1, 0, b2 - b1, segHeight);
            // 구간 3: 원본 비디오의 2*segWidth ~ 3*segWidth 구간의 b3-b2만큼 보여짐
            mergeContext.drawImage(vid, segWidth * 2, 0, segWidth, segHeight, b2, 0, b3 - b2, segHeight);
            // 구간 4: 원본 비디오의 3*segWidth ~ 4*segWidth 구간의 b4-b3만큼 보여짐
            mergeContext.drawImage(vid, segWidth * 3, 0, segWidth, segHeight, b3, 0, b4 - b3, segHeight);
            // 구간 5: 원본 비디오의 4*segWidth ~ 5*segWidth 구간의 나머지 부분
            mergeContext.drawImage(vid, segWidth * 4, 0, segWidth, segHeight, b4, 0, segWidth - b4, segHeight);
          
            // 경계선을 그리기 (네 개)
            mergeContext.beginPath();
            mergeContext.moveTo(b1, 0);
            mergeContext.lineTo(b1, segHeight);
            mergeContext.moveTo(b2, 0);
            mergeContext.lineTo(b2, segHeight);
            mergeContext.moveTo(b3, 0);
            mergeContext.lineTo(b3, segHeight);
            mergeContext.moveTo(b4, 0);
            mergeContext.lineTo(b4, segHeight);
            mergeContext.strokeStyle = "#AAAAAA";
            mergeContext.lineWidth = 5;
            mergeContext.stroke();
          
            // handle (경계) 표시: 각 handle마다 원과 화살표를 그림  
            // handle들의 수평 위치는 b1, b2, b3, b4이고, 세로 위치는 handleYs 배열로 지정 (예시: [10%, 30%, 70%, 90%] of segHeight)
            var handleYs = [segHeight * 0.1, segHeight * 0.3, segHeight * 0.7, segHeight * 0.9];
            var boundaries = [b1, b2, b3, b4];
          
            for (var i = 0; i < boundaries.length; i++) {
                var cx = boundaries[i];
                var cy = handleYs[i];
                // 원 그리기 (handle)
                mergeContext.beginPath();
                mergeContext.arc(cx, cy, circleRadius, 0, 2 * Math.PI, false);
                mergeContext.fillStyle = "#FFD79340";
                mergeContext.fill();
                mergeContext.closePath();
          
                // 화살표 그리기 (간단한 삼각형)
                mergeContext.beginPath();
                mergeContext.moveTo(cx, cy - 10);
                mergeContext.lineTo(cx - 10, cy + 10);
                mergeContext.lineTo(cx + 10, cy + 10);
                mergeContext.closePath();
                mergeContext.fillStyle = "#AAAAAA";
                mergeContext.fill();
            }
          
            requestAnimationFrame(drawLoop);
          }
          
          requestAnimationFrame(drawLoop);
    }
}

Number.prototype.clamp = function(min, max) {
    return Math.min(Math.max(this, min), max);
};


function resizeAndPlay5(element) {
    var cv = document.getElementById(element.id + "Merge");
    cv.width = element.videoWidth / 5;
    cv.height = element.videoHeight;
    element.play();
    element.style.height = "0px";  // Hide video without stopping it
    playVids5(element.id);
}
