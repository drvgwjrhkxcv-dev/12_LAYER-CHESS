import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI()

# --- [1] 인메모리 회원 관리 및 로그인 시스템 ---
# 실제 운영시에는 DB에 연동하지만, 채널원 테스트용으로 메모리에 저장 관리합니다.
USER_DB = {
    "admin": "admin123",
    "GJKJKA": "chess12",
    "player1": "1234",
    "player2": "1234"
}
ACTIVE_SESSIONS: Dict[str, str] = {} # cookie_token: username

# --- [2] 12평면체스 게임 규칙 엔진 (온라인 세션 독립형) ---
class TwelveLayerChessGame:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.white_player: Optional[str] = None
        self.black_player: Optional[str] = None
        self.max_planes = 12
        self.opened_planes = 6
        self.current_turn = "W"
        self.turn_history = {i: False for i in range(self.opened_planes)}
        self.pioneer_clicks = {"W": 6, "B": 6}
        self.boards = {i: [[None for _ in range(8)] for _ in range(8)] for i in range(self.max_planes)}
        self.setup_all_initial_layers()

    def setup_all_initial_layers(self):
        for p in range(4): self._fill_standard_layer(p)
        self._replace_piece(1, "Rook", "Transformer")
        self._replace_piece(2, "Bishop", "Hook")
        self._replace_piece(3, "Queen", "Transformer")
        
        self._fill_standard_layer(4)
        self._replace_piece(4, "Queen", "Dragon")
        self.boards[4][1][2] = "W_Magnet"; self.boards[4][1][5] = "W_Magnet"
        self.boards[4][6][2] = "B_Magnet"; self.boards[4][6][5] = "B_Magnet"
        self.boards[4][1][4] = "W_Pioneer"; self.boards[4][6][4] = "B_Pioneer"

        for c in range(8):
            self.boards[5][1][c] = "W_Magnet"
            self.boards[5][6][c] = "B_Magnet"
        self._replace_piece(5, "Bishop", "Hook")
        self._replace_piece(5, "Knight", "Pioneer")
        self._replace_piece(5, "Queen", "Transformer")
        self._replace_piece(5, "Rook", "Dragon")

    def _fill_standard_layer(self, p):
        back = ["Rook", "Knight", "Bishop", "Queen", "King", "Bishop", "Knight", "Rook"]
        for c in range(8):
            self.boards[p][0][c] = f"W_{back[c]}"
            self.boards[p][1][c] = "W_Pawn"
            self.boards[p][6][c] = "B_Pawn"
            self.boards[p][7][c] = f"B_{back[c]}"

    def _replace_piece(self, p, old_name, new_name):
        for r in range(8):
            for c in range(8):
                if self.boards[p][r][c] and old_name in self.boards[p][r][c]:
                    color = self.boards[p][r][c].split("_")[0]
                    self.boards[p][r][c] = f"{color}_{new_name}"

    def _get_binary_mask_5x5(self, plane, r, c):
        mask = np.zeros((5, 5), dtype=int)
        target_plane = plane
        if plane >= self.opened_planes or plane < 0: target_plane = 0 
        for i in range(5):
            for j in range(5):
                tr, tc = r - 2 + i, c - 2 + j
                if 0 <= tr < 8 and 0 <= tc < 8:
                    if self.boards[target_plane][tr][tc] is not None: mask[i][j] = 1
        return mask

    def get_legal_moves(self, p, r, c):
        if self.turn_history.get(p, False): return []
        piece = self.boards[p][r][c]
        if not piece or not piece.startswith(self.current_turn): return []
        color, p_type = piece.split("_")
        moves = []
        is_light = (r + c) % 2 == 0

        if p_type == "Hook":
            offsets = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)] if is_light else [(-1,0),(1,0),(0,-1),(0,1)]
            if is_light:
                for dr, dc in offsets:
                    tr, tc = r + dr, c + dc
                    if 0 <= tr < 8 and 0 <= tc < 8:
                        target = self.boards[p][tr][tc]
                        if not target or not target.startswith(color): moves.append({"p": p, "r": tr, "c": tc})
            else:
                for dr, dc in offsets:
                    tr, tc = r + dr, c + dc
                    while 0 <= tr < 8 and 0 <= tc < 8:
                        target = self.boards[p][tr][tc]
                        if not target: moves.append({"p": p, "r": tr, "c": tc})
                        elif not target.startswith(color): moves.append({"p": p, "r": tr, "c": tc}); break
                        else: break
                        tr += dr; tc += dc
            for i in range(5):
                for j in range(5):
                    tr, tc = r - 2 + i, c - 2 + j
                    if 0 <= tr < 8 and 0 <= tc < 8 and (tr != r or tc != c):
                        if self.boards[p][tr][tc]: moves.append({"p": p, "r": tr, "c": tc, "action": "swap"})

        elif p_type == "Transformer":
            A = self._get_binary_mask_5x5(p, r, c)
            B = self._get_binary_mask_5x5(p + 1, r, c)
            G = A @ B
            for i in range(5):
                for j in range(5):
                    tr, tc = r - 2 + i, c - 2 + j
                    if 0 <= tr < 8 and 0 <= tc < 8:
                        if (is_light and G[i][j] == 1) or (not is_light and G[i][j] == 0):
                            target = self.boards[p][tr][tc]
                            if not target or not target.startswith(color): moves.append({"p": p, "r": tr, "c": tc})

        elif p_type == "Dragon":
            A = self._get_binary_mask_5x5(p - 1, r, c)
            B = self._get_binary_mask_5x5(p + 1, r, c)
            C = np.dot(A, B)
            F_left = np.zeros((5, 3), dtype=int)
            F_right = np.zeros((3, 5), dtype=int)
            for i in range(5):
                for j in range(3):
                    if 0 <= r-2+i < 8 and 0 <= c-1+j < 8 and self.boards[p][r-2+i][c-1+j]: F_left[i][j] = 1
            for i in range(3):
                for j in range(5):
                    if 0 <= r-1+i < 8 and 0 <= c-2+j < 8 and self.boards[p][r-1+i][c-2+j]: F_right[i][j] = 1
            F = np.dot(F_left, F_right)
            G = C @ F
            for i in range(5):
                for j in range(5):
                    tr, tc = r - 2 + i, c - 2 + j
                    if 0 <= tr < 8 and 0 <= tc < 8:
                        if (is_light and G[i][j] == 1) or (not is_light and G[i][j] == 0):
                            target = self.boards[p][tr][tc]
                            if not target or not target.startswith(color): moves.append({"p": p, "r": tr, "c": tc})
            for tp in [p-1, p+1]:
                if 0 <= tp < self.opened_planes and not self.boards[tp][r][c]: moves.append({"p": tp, "r": r, "c": c})

        elif p_type == "Pioneer":
            ko = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
            for dr, dc in ko:
                tr, tc = r + dr, c + dc
                if 0 <= tr < 8 and 0 <= tc < 8:
                    target = self.boards[p][tr][tc]
                    if not target or not target.startswith(color): moves.append({"p": p, "r": tr, "c": tc, "action": "move"})
            if self.pioneer_clicks[color] > 0 and self.opened_planes < self.max_planes:
                moves.append({"p": self.opened_planes, "r": r, "c": c, "action": "forge"})
        else:
            offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            for dr, dc in offsets:
                tr, tc = r + dr, c + dc
                if 0 <= tr < 8 and 0 <= tc < 8:
                    target = self.boards[p][tr][tc]
                    if not target or not target.startswith(color): moves.append({"p": p, "r": tr, "c": tc})
        return moves

    def execute_move(self, fp, fr, fc, tp, tr, tc, action=None):
        piece = self.boards[fp][fr][fc]
        if not piece: return False
        color = piece.split("_")[0]

        if action == "swap":
            self.boards[fp][fr][fc], self.boards[tp][tr][tc] = self.boards[tp][tr][tc], self.boards[fp][fr][fc]
            self.turn_history[fp] = True
            self._check_and_switch_turn()
            return True

        if action == "forge":
            self.pioneer_clicks[color] -= 1
            tp_idx = self.opened_planes
            self._fill_standard_layer(tp_idx)
            for c in range(8):
                self.boards[tp_idx][1][c] = "W_Magnet"
                self.boards[tp_idx][6][c] = "B_Magnet"
            self._replace_piece(tp_idx, "Bishop", "Hook")
            self._replace_piece(tp_idx, "Knight", "Pioneer")
            self._replace_piece(tp_idx, "Queen", "Transformer")
            self._replace_piece(tp_idx, "Rook", "Dragon")
            self.opened_planes += 1
            self.turn_history[fp] = True
            self._check_and_switch_turn()
            return True

        self.boards[fp][fr][fc] = None
        self.boards[tp][tr][tc] = piece
        self.turn_history[fp] = True
        self._check_and_switch_turn()
        return True

    def _check_and_switch_turn(self):
        if all(self.turn_history[p] for p in range(self.opened_planes)):
            self.current_turn = "B" if self.current_turn == "W" else "W"
            self.turn_history = {i: False for i in range(self.opened_planes)}

    def get_match_verdict(self):
        total_sum = 0
        plane_scores = {}
        for p in range(self.opened_planes):
            w_king = any("W_King" in str(self.boards[p][r][c]) for r in range(8) for c in range(8))
            b_king = any("B_King" in str(self.boards[p][r][c]) for r in range(8) for c in range(8))
            status = 0
            if not b_king: status = 1
            elif not w_king: status = -1
            plane_scores[p] = status
            total_sum += status
        return {"scores": plane_scores, "total": total_sum, "verdict": "White Win" if total_sum > 0 else ("Black Win" if total_sum < 0 else "Playing")}

    def to_json(self):
        return {
            "boards": {p: self.boards[p] for p in range(self.opened_planes)},
            "opened_planes": self.opened_planes,
            "turn": self.current_turn,"pioneer_clicks": self.pioneer_clicks,"turn_history": self.turn_history,"white_player": self.white_player,"black_player": self.black_player,"verdict_data": self.get_match_verdict()}
ROOMS: Dict[str, TwelveLayerChessGame] = {}
CONNECTIONS: Dict[str, List[WebSocket]] = {}
async def broadcast_room(room_id: str):
    if room_id in ROOMS and room_id in CONNECTIONS:
        payload = json.dumps({"type": "sync", "data": ROOMS[room_id].to_json()})
        for ws in CONNECTIONS[room_id]:
            try: await ws.send_text(payload)
            except: pass
class LoginModel(BaseModel):
    username: str
    password: str
@app.post("/api/login")
def login(data: LoginModel, response: Response):
    if USER_DB.get(data.username) == data.password:
        token = f"token_{data.username}"
        ACTIVE_SESSIONS[token] = data.username
        response.set_cookie(key="session_token", value=token, httponly=True)
        return {"status": "success", "username": data.username}
    raise HTTPException(status_code=401, detail="아이디 혹은 비밀번호가 틀렸습니다.")
@app.get("/api/user")
def get_user(request: Request):
    token = request.cookies.get("session_token")
    if token in ACTIVE_SESSIONS:
        return {"logged_in": True, "username": ACTIVE_SESSIONS[token]}
    return {"logged_in": False}
@app.post("/api/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get("session_token")
    if token in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[token]
        response.delete_cookie("session_token")
        return {"status": "success"}
@app.get("/api/rooms")
def list_rooms():
    return {"rooms": [{ "id": rid, "w": r.white_player, "b": r.black_player } for rid, r in ROOMS.items()]}
@app.post("/api/room/create")
def create_room(room_id: str):
    if room_id in ROOMS:
        raise HTTPException(status_code=400, detail="이미 존재하는 방 ID입니다.")
    ROOMS[room_id] = TwelveLayerChessGame(room_id)
    return {"status": "created"}

@app.websocket("/ws/game/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, user: str):
    await websocket.accept()
    if room_id not in ROOMS:
        ROOMS[room_id] = TwelveLayerChessGame(room_id)
        game = ROOMS[room_id]# 흑/백 플레이어 자동 배정 시스템
        if not game.white_player and game.black_player != user: game.white_player = user
        elif not game.black_player and game.white_player != user: game.black_player = user
        if room_id not in CONNECTIONS:
            CONNECTIONS[room_id] = []
            CONNECTIONS[room_id].append(websocket)
            await broadcast_room(room_id)
            try:
                while True:
                    text_data = await websocket.receive_text()
                    req = json.loads(text_data)
                    if req["type"] == "get_moves":
                        moves = game.get_legal_moves(req["p"], req["r"], req["c"])
                        await websocket.send_text(json.dumps({"type": "moves", "moves": moves}))
                    elif req["type"] == "move":# 대국 턴 권한 검증 및 제어
                        player_color = "W" if game.white_player == user else ("B" if game.black_player == user else None)
                        if player_color != game.current_turn:
                            continue # 내 턴이 아니면 연산 무시
                        success = game.execute_move(req["fp"], req["fr"], req["fc"], req["tp"], req["tr"], req["tc"], req.get("action"))
                        if success:
                            await broadcast_room(room_id)
            except WebSocketDisconnect:
                    CONNECTIONS[room_id].remove(websocket)
                    await broadcast_room(room_id)
@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("index.html", "r", encoding="utf-8") as f: return f.read()
