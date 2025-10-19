// Global state
let demoBoard, analysisBoard;
let demoGame, analysisGame;
let demoInterval;
let uploadedPGN = null;
let analysisData = null;
let currentMoveIndex = 0;
let moveHistory = [];

// Color thresholds
const THRESHOLDS = {
    low: 0.4,
    high: 0.7
};

// Util to get color based on probability
function getColorForProb(prob) {
    if (prob < THRESHOLDS.low) return '#10b981'; // g
    if (prob < THRESHOLDS.high) return '#f59e0b'; // y
    return '#ef4444'; // r
}

// Initialize demo board
function initDemoBoard() {
    demoBoard = Chessboard('demo-board', {
        position: 'start',
        draggable: false,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    });
    demoGame = new Chess();

    fetchDemoGame();
}

async function fetchDemoGame() {
    try {
        const response = await fetch('/api/demo-game');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        const pgn = data.pgn;
        
        demoGame.load_pgn(pgn);
        playDemoGame();
    } catch (error) {
        console.error('Error fetching demo game:', error);
        
        // Fallback to mock PGN if endpoint fails
        const mockPGN = '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O';
        demoGame.load_pgn(mockPGN);
        playDemoGame();
    }
}

function playDemoGame() {
    demoGame.reset();
    demoBoard.position('start');

    const moves = demoGame.history({ verbose: true });
    demoGame.reset();

    let moveIndex = 0;

    demoInterval = setInterval(() => {
        if (moveIndex < moves.length) {
            demoGame.move(moves[moveIndex]);
            demoBoard.position(demoGame.fen());
            moveIndex++;
        } else {
            // Loop back to start
            demoGame.reset();
            demoBoard.position('start');
            moveIndex = 0;
        }
    }, 1000);
}

async function analyzeGame(pgn) {
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ pgn: pgn })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Transform backend format to frontend format
        // data[0] = white, data[1] = black
        return {
            white: {
                overallProbability: data[0].score / 100, // Convert to 0-1 range
                moveProbs: data[0].moves.map(m => m.prob / 100)
            },
            black: {
                overallProbability: data[1].score / 100,
                moveProbs: data[1].moves.map(m => m.prob / 100)
            }
        };
    } catch (error) {
        console.error('Error analyzing game:', error);
        throw error; // Re-throw so caller can handle it
    }
}

function initAnalysisBoard() {
    analysisBoard = Chessboard('analysis-board', {
        position: 'start',
        draggable: false,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    });

    analysisGame = new Chess();

    try {
        const loadResult = analysisGame.load_pgn(uploadedPGN);
        if (!loadResult) {
            alert('Error loading PGN for analysis');
            return;
        }
    } catch (error) {
        alert('Error loading PGN: ' + error.message);
        return;
    }

    // Build move history
    moveHistory = [];
    const tempGame = new Chess();
    tempGame.load_pgn(uploadedPGN);
    const moves = tempGame.history({ verbose: true });
    tempGame.reset();

    moveHistory.push({ fen: tempGame.fen(), move: null });
    moves.forEach(move => {
        tempGame.move(move);
        moveHistory.push({
            fen: tempGame.fen(),
            move: move,
            from: move.from,
            to: move.to
        });
    });

    // Display overall probabilities
    document.getElementById('white-prob').textContent =
        (analysisData.white.overallProbability * 100).toFixed(1) + '%';
    document.getElementById('black-prob').textContent =
        (analysisData.black.overallProbability * 100).toFixed(1) + '%';

    // Populate move list
    populateMoveList();

    // Set initial position
    currentMoveIndex = 0;
    updatePosition();
}

function populateMoveList() {
    const moveListEl = document.getElementById('move-list');
    moveListEl.innerHTML = '';

    const game = new Chess();
    game.load_pgn(uploadedPGN);
    const moves = game.history();

    for (let i = 0; i < moves.length; i += 2) {
        const moveNum = Math.floor(i / 2) + 1;
        const whiteMove = moves[i];
        const blackMove = moves[i + 1] || '';

        const whiteProb = analysisData.white.moveProbs[Math.floor(i / 2)];
        const blackProb = i + 1 < moves.length ? analysisData.black.moveProbs[Math.floor(i / 2)] : null;

        const moveDiv = document.createElement('div');
        moveDiv.className = 'flex gap-2 items-center';

        moveDiv.innerHTML = `
            <span class="text-gray-600 font-semibold w-8">${moveNum}.</span>
            <span class="flex-1 px-3 py-2 rounded move-item cursor-pointer" 
                  style="background-color: ${getColorForProb(whiteProb)}; color: white;"
                  data-move-index="${i + 1}">
                ${whiteMove}
            </span>
            ${blackMove ? `
                <span class="flex-1 px-3 py-2 rounded move-item cursor-pointer" 
                      style="background-color: ${getColorForProb(blackProb)}; color: white;"
                      data-move-index="${i + 2}">
                    ${blackMove}
                </span>
            ` : '<span class="flex-1"></span>'}
        `;

        moveListEl.appendChild(moveDiv);
    }

    // Add click handlers
    document.querySelectorAll('.move-item').forEach(item => {
        item.addEventListener('click', (e) => {
            const index = parseInt(e.target.dataset.moveIndex);
            currentMoveIndex = index;
            updatePosition();
        });
    });
}

function updatePosition() {
    if (currentMoveIndex < 0 || currentMoveIndex >= moveHistory.length) return;

    const position = moveHistory[currentMoveIndex];
    analysisBoard.position(position.fen);

    // Clear previous arrow
    const arrowOverlay = document.getElementById('arrow-overlay');
    if (arrowOverlay) {
        arrowOverlay.innerHTML = '';
    }

    // Draw arrow if not initial position
    if (position.move && arrowOverlay) {
        drawArrow(position.from, position.to);
    }

    // Update current move info
    if (currentMoveIndex === 0) {
        document.getElementById('current-player').textContent = '-';
        document.getElementById('current-prob').textContent = '-';
        document.getElementById('current-prob').style.color = '';
    } else {
        const isWhiteMove = (currentMoveIndex - 1) % 2 === 0;
        const moveIndex = Math.floor((currentMoveIndex - 1) / 2);
        const prob = isWhiteMove ?
            analysisData.white.moveProbs[moveIndex] :
            analysisData.black.moveProbs[moveIndex];

        document.getElementById('current-player').textContent = isWhiteMove ? 'White' : 'Black';
        document.getElementById('current-prob').textContent = (prob * 100).toFixed(1) + '%';
        document.getElementById('current-prob').style.color = getColorForProb(prob);
    }

    // Highlight active move in list
    document.querySelectorAll('.move-item').forEach(item => {
        item.classList.remove('ring-4', 'ring-blue-300');
        if (parseInt(item.dataset.moveIndex) === currentMoveIndex) {
            item.classList.add('ring-4', 'ring-blue-300');
        }
    });
}

function drawArrow(from, to) {
    const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const fromFile = files.indexOf(from[0]);
    const fromRank = 8 - parseInt(from[1]);
    const toFile = files.indexOf(to[0]);
    const toRank = 8 - parseInt(to[1]);

    const squareSize = 700 / 8;
    const x1 = fromFile * squareSize + squareSize / 2;
    const y1 = fromRank * squareSize + squareSize / 2;
    const x2 = toFile * squareSize + squareSize / 2;
    const y2 = toRank * squareSize + squareSize / 2;

    const svg = document.getElementById('arrow-overlay');

    // Calculate arrow angle
    const angle = Math.atan2(y2 - y1, x2 - x1);

    // Shorten arrow to not cover destination square
    const arrowHeadSize = 10;
    const shortenBy = squareSize * 0.5 + (arrowHeadSize/2);
    const x2Short = x2 - (shortenBy + arrowHeadSize) * Math.cos(angle);
    const y2Short = y2 - (shortenBy + arrowHeadSize) * Math.sin(angle);

    // Create arrow line
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x1);
    line.setAttribute('y1', y1);
    line.setAttribute('x2', x2Short);
    line.setAttribute('y2', y2Short);
    line.setAttribute('stroke', '#3b82f6');
    line.setAttribute('stroke-width', '4');
    line.setAttribute('stroke-opacity', ".5");
    line.setAttribute('marker-end', 'url(#arrowhead)');

    // Create arrowhead marker
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', 'arrowhead');
    marker.setAttribute('markerWidth', '10');
    marker.setAttribute('markerHeight', '10');
    marker.setAttribute('refX', '5');
    marker.setAttribute('refY', '3');
    marker.setAttribute('orient', 'auto');

    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('points', '0 0, 10 3, 0 6');
    polygon.setAttribute('fill', '#3b82f6');
    polygon.setAttribute('fill-opacity', '0.5');

    marker.appendChild(polygon);
    defs.appendChild(marker);
    svg.appendChild(defs);
    svg.appendChild(line);
}

// Navigation controls
document.addEventListener('DOMContentLoaded', () => {
    initDemoBoard();

    // File upload handling
    const uploadZone = document.getElementById('upload-zone');
    if (uploadZone) {
        uploadZone.addEventListener('click', () => {
            document.getElementById('file-input').click();
        });
    }

    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];

            if (!file) {
                alert('Please select a PGN file');
                return;
            }

            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    console.log(event.target.result);

                    // Validate PGN
                    const testGame = new Chess();
                    const loadResult = testGame.load_pgn(event.target.result);

                    if (!loadResult) {
                        alert('Invalid PGN format');
                        uploadedPGN = null;
                        // Disable analyze button
                        const analyzeBtn = document.getElementById('analyze-btn');
                        analyzeBtn.disabled = true;
                        analyzeBtn.classList.add('bg-gray-300', 'text-gray-500', 'cursor-not-allowed');
                        analyzeBtn.classList.remove('bg-blue-500', 'text-white', 'hover:bg-blue-600', 'cursor-pointer');
                        return;
                    }

                    // If we get here, PGN is valid
                    uploadedPGN = event.target.result;

                    // Update UI
                    document.getElementById('file-name').textContent = file.name;
                    document.getElementById('file-name').classList.remove('hidden');

                    const analyzeBtn = document.getElementById('analyze-btn');
                    analyzeBtn.disabled = false;
                    analyzeBtn.classList.remove('bg-gray-300', 'text-gray-500', 'cursor-not-allowed');
                    analyzeBtn.classList.add('bg-blue-500', 'text-white', 'hover:bg-blue-600', 'cursor-pointer');

                    console.log('PGN loaded successfully');
                } catch (error) {
                    console.log(error);
                    alert('Invalid PGN format');
                    uploadedPGN = null;
                    // Disable analyze button
                    const analyzeBtn = document.getElementById('analyze-btn');
                    analyzeBtn.disabled = true;
                    analyzeBtn.classList.add('bg-gray-300', 'text-gray-500', 'cursor-not-allowed');
                    analyzeBtn.classList.remove('bg-blue-500', 'text-white', 'hover:bg-blue-600', 'cursor-pointer');
                }
            };
            reader.readAsText(file);
        });
    }

    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            if (!uploadedPGN) {
                alert('Please upload a valid PGN file first');
                return;
            }

            // Validate PGN one more time before analysis
            try {
                const testGame = new Chess();
                const loadResult = testGame.load_pgn(uploadedPGN);
                if (!loadResult) {
                    alert('Invalid PGN format. Please upload a valid PGN file.');
                    return;
                }
            } catch (error) {
                alert('Invalid PGN format. Please upload a valid PGN file.');
                console.error(error);
                return;
            }

            // Stop demo animation
            clearInterval(demoInterval);

            try {
                // Call backend instead of mock
                analysisData = await analyzeGame(uploadedPGN);

                if (!analysisData) {
                    alert('Error analyzing game - no data returned');
                    return;
                }

                // Switch views
                document.getElementById('state-upload').classList.add('hidden');
                document.getElementById('state-analysis').classList.remove('hidden');

                // Initialize analysis board
                initAnalysisBoard();
            } catch (error) {
                alert('Error analyzing game. Please try again.');
                console.error(error);
            }
        });
    }

    const prevBtn = document.getElementById('prev-btn');
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentMoveIndex > 0) {
                currentMoveIndex--;
                updatePosition();
            }
        });
    }

    const nextBtn = document.getElementById('next-btn');
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            if (currentMoveIndex < moveHistory.length - 1) {
                currentMoveIndex++;
                updatePosition();
            }
        });
    }

    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            currentMoveIndex = 0;
            updatePosition();
        });
    }
});