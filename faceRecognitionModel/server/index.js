const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");

const app = express();

app.use(cors());

const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: "*"
  }
});

let playersOnline = 0;

io.on("connection", (socket) => {

  playersOnline++;

  io.emit(
    "playersOnline",
    playersOnline
  );

  console.log(
    "User connected:",
    socket.id
  );

  // JOIN ROOM
  socket.on(
    "joinRoom",
    ({ roomId, username }) => {

      socket.join(roomId);

      const room =
        io.sockets.adapter.rooms.get(
          roomId
        );

      const playersInRoom =
        room ? room.size : 0;

      let role =
        "Spectator";

      if (playersInRoom === 1) {
        role = "White";
      }
      else if (
        playersInRoom === 2
      ) {
        role = "Black";
      }

      socket.emit(
        "playerRole",
        role
      );

      io.to(roomId).emit(
        "roomMessage",
        `${username} joined room ${roomId} as ${role}`
      );

      console.log(
        `${username} joined ${roomId} as ${role}`
      );

    }
  );

  // MOVE EVENT
  socket.on("move", (data) => {

    io.to(data.roomId)
      .emit("move", data.move);

});

  socket.on(
    "disconnect",
    () => {

      playersOnline--;

      io.emit(
        "playersOnline",
        playersOnline
      );

      console.log(
        "User disconnected"
      );

    }
  );

});

server.listen(
  3001,
  () => {

    console.log(
      "Server running on port 3001"
    );
  
}
);