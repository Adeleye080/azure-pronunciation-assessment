import os
import json
import uuid
import time
import threading
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk

# For chart generation
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np  # For numerical operations with phonemes if needed

app = Flask(__name__)
app.config["SECRET_KEY"] = (
    "your_secret_key_here"  # IMPORTANT: Change this in production
)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Azure Speech Service Configuration (use environment variables in production)
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "YOUR_AZURE_SPEECH_KEY_HERE")
AZURE_SPEECH_REGION = os.environ.get(
    "AZURE_SPEECH_REGION", "eastus"
)  # Replace with your actual region, e.g., "eastus"

# Store results for asynchronous retrieval
assessment_results = {}
job_status = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/assess", methods=["POST"])
def assess():
    if "audio_file" not in request.files:
        flash("No file part", "danger")
        return jsonify({"error": "No file part"}), 400

    file = request.files["audio_file"]
    reference_text = request.form.get(
        "reference_text", "The quick brown fox jumps over the lazy dog."
    )
    language = request.form.get("language", "en-US")

    if file.filename == "":
        flash("No selected file", "danger")
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        job_id = unique_id  # Use the unique ID as job_id
        original_filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(original_filepath)
        print(f"File saved to: {original_filepath} for job ID: {job_id}")

        job_status[job_id] = {
            "status": "processing",
            "message": "Assessment is processing...",
        }

        # Start background thread for assessment
        thread = threading.Thread(
            target=background_assessment_task,
            args=(job_id, original_filepath, reference_text, language),
        )
        thread.start()

        # Respond immediately to the frontend
        return jsonify(
            {
                "job_id": job_id,
                "message": "Assessment started. Polling for results is recommended.",
            }
        )

    return jsonify({"error": "An unexpected error occurred."}), 500


@app.route("/get_results/<job_id>", methods=["GET"])
def get_results(job_id):
    if job_id in job_status:
        if job_status[job_id]["status"] == "completed":
            results = assessment_results.pop(job_id, None)  # Retrieve and remove
            results.update({"status": "completed"})
            job_status.pop(job_id, None)  # Remove status
            return jsonify(results)
        elif job_status[job_id]["status"] == "failed":
            error_message = job_status[job_id]["message"]
            job_status.pop(job_id, None)  # Remove status
            return jsonify({"status": "failed", "error": error_message}), 500
        else:
            return (
                jsonify(
                    {"status": "processing", "message": job_status[job_id]["message"]}
                ),
                202,
            )  # Accepted (still processing)
    else:
        return (
            jsonify(
                {
                    "status": "not_found",
                    "message": "Job ID not found or already retrieved.",
                }
            ),
            404,
        )


def background_assessment_task(job_id, original_filepath, reference_text, language):
    print(f"[{job_id}] Extracting audio from {original_filepath}...")
    temp_dir = None
    converted_audio_path = None
    try:
        # Create a temporary directory for converted audio
        temp_dir = os.path.join("/tmp", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        converted_audio_path = os.path.join(temp_dir, "Recording.wav")

        # Convert to WAV using pydub (relies on ffmpeg)
        audio = AudioSegment.from_file(original_filepath)
        audio.export(converted_audio_path, format="wav")
        print(f"Audio extracted and converted to: {converted_audio_path}")

        # Send to Azure for pronunciation assessment
        print(f"[{job_id}] Sending audio to Azure for pronunciation assessment...")
        azure_result = assess_pronunciation(
            converted_audio_path, reference_text, language
        )

        if "error" in azure_result:
            job_status[job_id] = {
                "status": "failed",
                "message": f"Assessment failed: {azure_result['error']}",
            }
            print(f"[{job_id}] Assessment failed: {azure_result['error']}")
        else:
            # Process Azure results and generate charts
            processed_results = process_azure_results(azure_result)
            assessment_results[job_id] = processed_results
            job_status[job_id] = {
                "status": "completed",
                "message": "Assessment complete. Results stored.",
            }
            print(f"[{job_id}] Assessment complete. Results stored.")

    except Exception as e:
        error_msg = f"An unexpected error occurred during background assessment: {e}"
        job_status[job_id] = {"status": "failed", "message": error_msg}
        print(f"[{job_id}] {error_msg}")
    finally:
        # Clean up uploaded file
        if os.path.exists(original_filepath):
            os.remove(original_filepath)
            print(f"[{job_id}] Cleaned up uploaded file: {original_filepath}")
        # Clean up temporary audio and directory
        if converted_audio_path and os.path.exists(converted_audio_path):
            os.remove(converted_audio_path)
        if temp_dir and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
            print(f"[{job_id}] Cleaned up temporary audio and directory: {temp_dir}")


def assess_pronunciation(audio_file_path, reference_text, language):
    """
    Performs pronunciation assessment on an audio file using Azure Speech Service.
    Returns a dictionary containing the pronunciation assessment results, or None if an error occurs.
    """
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        return {
            "error": "Azure Speech Service key or region not configured. Please check server setup."
        }

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION
    )
    speech_config.speech_recognition_language = language

    pronunciation_config_json = {
        "referenceText": reference_text,
        "gradingSystem": "HundredMark",
        "granularity": "Phoneme",
        "enableMiscue": True,
    }

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        json_string=json.dumps(pronunciation_config_json)
    )

    audio_config = speechsdk.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )
    pronunciation_config.apply_to(speech_recognizer)

    print(f"Sending audio for assessment: {audio_file_path}")
    print(f"Reference text: {reference_text}")

    try:
        result = speech_recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Pronunciation assessment result received.")
            return json.loads(
                result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
            )
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized.")
            return {"error": "No speech could be recognized in the audio."}
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            error_msg = f"Speech Recognition canceled: {cancellation_details.reason}"
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                error_msg += f". Error details: {cancellation_details.error_details}"
            print(error_msg)
            return {"error": error_msg}
    except Exception as e:
        print(f"Error during pronunciation assessment: {e}")
        return {"error": f"An error occurred during assessment: {e}"}


def process_azure_results(azure_raw_result):
    """
    Processes the raw Azure Speech Service result to extract relevant scores
    and generate charts.
    """
    if not azure_raw_result or azure_raw_result.get("RecognitionStatus") != "Success":
        return {"error": "Invalid or unsuccessful Azure result."}

    n_best = azure_raw_result["NBest"][0]
    pron_assessment = n_best["PronunciationAssessment"]
    words = n_best["Words"]

    # 1. Overall Scores
    overall_scores = {
        "accuracy_score": pron_assessment["AccuracyScore"],
        "fluency_score": pron_assessment["FluencyScore"],
        "completeness_score": pron_assessment["CompletenessScore"],
        "pronunciation_score": pron_assessment["PronScore"],
    }

    # 2. Statistics
    speech_duration_ms = azure_raw_result["Duration"]
    speech_duration_s = speech_duration_ms / 1_000_000  # Convert 100ns to seconds
    num_words_recognized = len(words)
    speaking_rate_wpm = (
        (num_words_recognized / speech_duration_s) * 60 if speech_duration_s > 0 else 0
    )

    statistics = {
        "speech_duration": speech_duration_s,
        "speaking_rate": speaking_rate_wpm,
    }

    recognized_text = n_best["Display"]

    # 3. Chart Generation
    charts = {}

    # Overall Scores Chart
    scores_labels = ["Accuracy", "Fluency", "Completeness", "Overall Pron."]
    scores_values = [
        overall_scores["accuracy_score"],
        overall_scores["fluency_score"],
        overall_scores["completeness_score"],
        overall_scores["pronunciation_score"],
    ]
    charts["overall_scores"] = create_bar_chart(
        scores_labels, scores_values, "Overall Scores", "Score", "Score (%)"
    )

    # Word-Level Accuracy Chart
    word_labels = [word["Word"] for word in words]
    word_accuracy_scores = [
        word["PronunciationAssessment"]["AccuracyScore"] for word in words
    ]
    charts["word_accuracy"] = create_bar_chart(
        word_labels, word_accuracy_scores, "Word-Level Accuracy", "Word", "Accuracy (%)"
    )

    # Speech Statistics Chart (Can be a bar chart or just display as text)
    # For now, let's make a simple bar chart if you want it visualized
    stat_labels = ["Speech Duration (s)", "Speaking Rate (WPM)"]
    stat_values = [statistics["speech_duration"], statistics["speaking_rate"]]
    charts["statistics_dashboard"] = create_bar_chart(
        stat_labels, stat_values, "Speech Statistics", "", "Value"
    )

    # Error Distribution (Mispronunciation, Omission, Insertion)
    error_types = {"Mispronunciation": 0, "Omission": 0, "Insertion": 0}
    for word in words:
        error_type = word["PronunciationAssessment"].get("ErrorType", "None")
        if error_type != "None":
            # Azure categorizes errors, but if 'None' is there, it's not an error.
            # We map specific error types here for simplicity.
            # 'Mispronunciation' typically covers accuracy issues.
            # 'Omission' and 'Insertion' are higher-level issues.
            # For simplicity, we'll count any non-None as a "Mispronunciation"
            # as per Azure's default grading. If you need fine-grained,
            # you'd need to parse the JSON more deeply for specific types.
            if error_type == "Mispronunciation":
                error_types["Mispronunciation"] += 1
            elif error_type == "Omission":
                error_types["Omission"] += 1
            elif error_type == "Insertion":
                error_types["Insertion"] += 1
            # Note: Azure's response for `ErrorType` can be more nuanced.
            # If `enableMiscue` is True, you might get 'Omission' or 'Insertion' at the word level.
            # If just accuracy is off, it might be 'Mispronunciation' or 'None' with low score.
            # For this example, we directly use the `ErrorType` provided.

    error_labels = list(error_types.keys())
    error_values = list(error_types.values())
    charts["error_distribution"] = create_pie_chart(
        error_labels, error_values, "Error Distribution"
    )

    # Phoneme Accuracy Heatmap (simplified example, can be much more complex)
    # This requires extracting phoneme-level data and creating a meaningful heatmap.
    # For simplicity, we'll show a bar chart of top/bottom phonemes, or a generic representation.
    # A true heatmap would require a 2D array of phonemes vs. features/scores.
    # Let's get phoneme accuracy scores for all unique phonemes that appeared
    phoneme_scores = {}
    for word in words:
        for syllable in word.get(
            "Syllables", []
        ):  # Iterate through syllables if available
            for phoneme_data in syllable.get("Phonemes", []):
                phoneme = phoneme_data["Phoneme"]
                score = phoneme_data["PronunciationAssessment"]["AccuracyScore"]
                if phoneme not in phoneme_scores:
                    phoneme_scores[phoneme] = []
                phoneme_scores[phoneme].append(score)

    avg_phoneme_scores = {p: np.mean(s) for p, s in phoneme_scores.items()}
    sorted_phonemes = sorted(avg_phoneme_scores.items(), key=lambda item: item[1])

    # Take top and bottom 5 for a representative chart if many phonemes
    if len(sorted_phonemes) > 10:
        display_phonemes = sorted_phonemes[:5] + sorted_phonemes[-5:]
    else:
        display_phonemes = sorted_phonemes

    phoneme_labels = [p[0] for p in display_phonemes]
    phoneme_accuracy_values = [p[1] for p in display_phonemes]

    charts["phoneme_heatmap"] = create_bar_chart(
        phoneme_labels,
        phoneme_accuracy_values,
        "Phoneme Accuracy (Avg. Scores)",
        "Phoneme",
        "Accuracy (%)",
    )

    return {
        "overall_scores": overall_scores,
        "statistics": statistics,
        "recognized_text": recognized_text,
        "charts": charts,
        "raw_azure_result": azure_raw_result,  # Optionally keep raw result for debug
    }


def create_bar_chart(labels, values, title, xlabel, ylabel):
    """Generates a bar chart and returns it as a base64 encoded PNG."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=values, palette="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 100)  # Scores are usually 0-100
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    plt.close()  # Close the plot to free up memory
    return img_b64


def create_pie_chart(labels, values, title):
    """Generates a pie chart and returns it as a base64 encoded PNG."""
    # Filter out labels with zero values to avoid issues with pie chart
    filtered_labels = [labels[i] for i, v in enumerate(values) if v > 0]
    filtered_values = [v for v in values if v > 0]

    if not filtered_values:  # Handle case where all values are zero
        plt.figure(figsize=(6, 6))
        plt.text(
            0.5,
            0.5,
            "No errors detected",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.axis("off")
    else:
        plt.figure(figsize=(8, 8))
        plt.pie(
            filtered_values,
            labels=filtered_labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette("pastel"),
        )
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title(title)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    plt.close()
    return img_b64


if __name__ == "__main__":
    app.run(debug=True)
