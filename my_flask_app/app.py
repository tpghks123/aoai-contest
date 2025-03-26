import os
import json
from datetime import datetime
from flask import Flask, render_template, request, Response, jsonify

# Whisper
import whisper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 임시 저장 (DB 대용)
AUDIO_TEXT_STORAGE = ""
USER_REQUEST_STORAGE = ""

# Whisper 모델 로딩 (최초 1회만)
model = whisper.load_model("small")  # "small", "medium", "large" 등

def whisper_speech_to_text(file_path):
    """
    Whisper로 음성을 텍스트로 변환
    file_path: 로컬 오디오 파일 경로
    return: 변환된 텍스트
    """
    # CPU 환경이면 fp16=False 필요. GPU 있으면 True 가능.
    result = model.transcribe(file_path, fp16=False)
    text = result["text"].strip()
    return text if text else "(인식된 텍스트가 없습니다.)"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global AUDIO_TEXT_STORAGE, USER_REQUEST_STORAGE

    audio_file = request.files.get("audio_file")
    user_request = request.form.get("org_info", "").strip()

    messages = []

    # 1) 녹음 파일
    if audio_file and audio_file.filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{audio_file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            audio_file.save(save_path)
            messages.append(f"오디오 파일 업로드 완료: {save_path}")
            # Whisper STT
            stt_result = whisper_speech_to_text(save_path)
            AUDIO_TEXT_STORAGE = stt_result
            messages.append(f"STT 변환 텍스트:\n{stt_result}")

        except Exception as e:
            messages.append(f"오디오 파일 처리 오류: {str(e)}")
    else:
        messages.append("오디오 파일이 선택되지 않았습니다.")

    # 2) 유저 요청 사항
    if len(user_request) < 1:
        messages.append("유저 요청사항이 비어있거나 너무 짧습니다.")
    else:
        USER_REQUEST_STORAGE = user_request
        messages.append("유저 요청사항 입력 완료:")
        messages.append(user_request)

    return render_template("result.html", messages=messages)

# (1) 변환된 텍스트 API
@app.route("/api/transcribed_text", methods=["GET"])
def get_transcribed_text():
    """
    어시웍스 에이전트에서 STT 결과 텍스트 가져가는 엔드포인트
    JSON 응답(유니코드 이스케이프 없이)
    """
    obj = {"transcribed_text": AUDIO_TEXT_STORAGE}

    # ensure_ascii=False 설정
    return Response(json.dumps(obj, ensure_ascii=False),
                    mimetype='application/json; charset=utf-8')

# (2) 유저 요청 사항 API
@app.route("/api/user_request", methods=["GET"])
def get_company_info():
    """
    어시웍스 에이전트에서 유저 요청사항을 가져가는 엔드포인트
    JSON 응답(유니코드 이스케이프 없이)
    """
    obj = {"user_request": USER_REQUEST_STORAGE}
    return Response(json.dumps(obj, ensure_ascii=False),
                    mimetype='application/json; charset=utf-8')

if __name__ == "__main__":
    app.run(debug=True, port=4040)
