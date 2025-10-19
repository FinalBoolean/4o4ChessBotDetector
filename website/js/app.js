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

function getColorForProb(prob) {
    if (prob < THRESHOLDS.low) return '#10b981'; // green
    if (prob < THRESHOLDS.high) return '#f59e0b'; // yellow
    return '#ef4444'; // red
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
    // TODO: Replace with actual endpoint
    // const response = await fetch('/api/demo-game');
    // const pgn = await response.text();
    
    // Mock PGN for demo
    const mockPGN = '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O';
    
    demoGame.load_pgn(mockPGN);
    playDemoGame();
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

function generateMockAnalysis(pgn) {
    const game = new Chess();
    game.load_pgn(pgn);
    const moves = game.history();
    
    // Generate random probabilities for demo
    const whiteMoves = [];
    const blackMoves = [];
    
    moves.forEach((move, i) => {
        const prob = Math.random();
        if (i % 2 === 0) {
            whiteMoves.push(prob);
        } else {
            blackMoves.push(prob);
        }
    });
    
    return {
        white: {
            overallProbability: whiteMoves.reduce((a, b) => a + b, 0) / whiteMoves.length,
            moveProbs: whiteMoves
        },
        black: {
            overallProbability: blackMoves.reduce((a, b) => a + b, 0) / blackMoves.length,
            moveProbs: blackMoves
        }
    };
}

function initAnalysisBoard() {
    analysisBoard = Chessboard('analysis-board', {
        position: 'start',
        draggable: false,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    });
    
    analysisGame = new Chess();
    analysisGame.load_pgn(uploadedPGN);
    
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
            <span class="flex-1 px-3 py-2 rounded move-item" 
                  style="background-color: ${getColorForProb(whiteProb)}; color: white;"
                  data-move-index="${i + 1}">
                ${whiteMove}
            </span>
            ${blackMove ? `
                <span class="flex-1 px-3 py-2 rounded move-item" 
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
    document.getElementById('arrow-overlay').innerHTML = '';
    
    // Draw arrow if not initial position
    if (position.move) {
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
        item.classList.remove('active');
        if (parseInt(item.dataset.moveIndex) === currentMoveIndex) {
            item.classList.add('active');
        }
    });
}

function drawArrow(from, to) {
    const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const fromFile = files.indexOf(from[0]);
    const fromRank = 8 - parseInt(from[1]);
    const toFile = files.indexOf(to[0]);
    const toRank = 8 - parseInt(to[1]);
    
    const squareSize = 500 / 8;
    const x1 = fromFile * squareSize + squareSize / 2;
    const y1 = fromRank * squareSize + squareSize / 2;
    const x2 = toFile * squareSize + squareSize / 2;
    const y2 = toRank * squareSize + squareSize / 2;
    
    const svg = document.getElementById('arrow-overlay');
    
    // Calculate arrow angle
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const arrowLength = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    
    // Shorten arrow to not cover destination square
    const shortenBy = squareSize * 0.3;
    const x2Short = x2 - shortenBy * Math.cos(angle);
    const y2Short = y2 - shortenBy * Math.sin(angle);
    
    // Create arrow line
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x1);
    line.setAttribute('y1', y1);
    line.setAttribute('x2', x2Short);
    line.setAttribute('y2', y2Short);
    line.setAttribute('stroke', '#3b82f6');
    line.setAttribute('stroke-width', '4');
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
    
    marker.appendChild(polygon);
    defs.appendChild(marker);
    svg.appendChild(defs);
    svg.appendChild(line);
}

// Navigation controls (keep these outside window.onload)
document.addEventListener('DOMContentLoaded', () => {
    initDemoBoard();
    
    // File upload handling
    document.getElementById('upload-zone').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });

    document.getElementById('file-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                uploadedPGN = event.target.result;
                document.getElementById('file-name').textContent = file.name;
                document.getElementById('file-name').classList.remove('hidden');
                
                const analyzeBtn = document.getElementById('analyze-btn');
                analyzeBtn.disabled = false;
                analyzeBtn.classList.remove('bg-gray-300', 'text-gray-500', 'cursor-not-allowed');
                analyzeBtn.classList.add('bg-blue-500', 'text-white', 'hover:bg-blue-600', 'cursor-pointer');
            };
            reader.readAsText(file);
        }
    });

    document.getElementById('analyze-btn').addEventListener('click', async () => {
        if (!uploadedPGN) return;
        clearInterval(demoInterval);
        analysisData = generateMockAnalysis(uploadedPGN);
        document.getElementById('state-upload').classList.add('hidden');
        document.getElementById('state-analysis').classList.remove('hidden');
        initAnalysisBoard();
    });

    document.getElementById('prev-btn').addEventListener('click', () => {
        if (currentMoveIndex > 0) {
            currentMoveIndex--;
            updatePosition();
        }
    });

    document.getElementById('next-btn').addEventListener('click', () => {
        if (currentMoveIndex < moveHistory.length - 1) {
            currentMoveIndex++;
            updatePosition();
        }
    });

    document.getElementById('reset-btn').addEventListener('click', () => {
        currentMoveIndex = 0;
        updatePosition();
    });
});