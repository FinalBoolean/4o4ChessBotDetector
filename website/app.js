import { Chess } from 'chess.js'
 const game = new Chess();

const board1 = Chessboard('board1', {
  position: 'start',
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
});

let moves = [];
let currentMoveIndex = 0;

function loadPGN() {
  const fileInput = document.getElementById('pgnFile');
  const file = fileInput.files[0];
  
  if (!file) {
    alert('Please select a PGN file');
    return;
  }
  
  const reader = new FileReader();
  reader.onload = function(e) {
    try{

        if (game.loadPgn(e.target.result,{ sloppy: true })) {
            moves = game.history();
            currentMoveIndex = 0;
            game.reset();
            board1.position('start');
            console.log(moves);
            alert('PGN loaded! Use Next/Previous buttons to navigate.');
        } else {
            alert('Invalid PGN format');
        }
    }catch(error){
        alert("Error handling PGN file " + error.message);
        console.log(error);
    }
  };
  reader.readAsText(file);
}

function nextMove() {
  if (currentMoveIndex < moves.length) {
    game.move(moves[currentMoveIndex]);
    board1.position(game.fen());
    currentMoveIndex++;
  }
}

function previousMove() {
  if (currentMoveIndex > 0) {
    game.undo();
    board1.position(game.fen());
    currentMoveIndex--;
  }
}

function resetGame() {
  game.reset();
  board1.position('start');
  currentMoveIndex = 0;
}

// Attach event listeners (replaces onclick attributes)
document.getElementById('loadBtn').addEventListener('click', loadPGN);
document.getElementById('prevBtn').addEventListener('click', previousMove);
document.getElementById('nextBtn').addEventListener('click', nextMove);
document.getElementById('resetBtn').addEventListener('click', resetGame);