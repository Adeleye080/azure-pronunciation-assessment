import os
import json
import uuid  # To generate unique job IDs
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk
import tempfile
import threading
from dotenv import load_dotenv
import time


load_dotenv()


# --- Configuration ---
# REPLACE WITH YOUR ACTUAL AZURE SPEECH SERVICE KEY AND REGION
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")  # e.g., "eastus", "westus"

app = Flask(__name__)
app.secret_key = "your_super_secret_key"  # Needed for flash messages
app.config["UPLOAD_FOLDER"] = "uploads"  # Temporary folder for uploaded files
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# In-memory store for assessment results.
# In a real-world app, use a database (Redis, PostgreSQL, etc.) for persistence and scalability.
ASSESSMENT_RESULTS = (
    {}
)  # Stores {job_id: {"status": "pending" | "complete" | "error", "data": {...}}}


# --- Helper Functions (Audio Processing) ---
def extract_audio(input_path):
    """
    Extracts audio from an input audio/video file and converts it to WAV format.
    Returns the path to the processed WAV audio file, or None if an error occurs.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        # Azure Speech Service generally prefers 16kHz, 16-bit, mono WAV
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        # Create a temporary file for the WAV output
        temp_dir = tempfile.mkdtemp()
        output_wav_path = os.path.join(
            temp_dir, os.path.basename(os.path.splitext(input_path)[0]) + ".wav"
        )

        audio.export(output_wav_path, format="wav")
        print(f"Audio extracted and converted to: {output_wav_path}")
        return output_wav_path
    except Exception as e:
        print(f"Error extracting audio from {input_path}: {e}")
        return None


# --- Helper Functions (Azure Speech Service Interaction) ---
def assess_pronunciation(audio_file_path, reference_text):
    """
    Performs pronunciation assessment on an audio file using Azure Speech Service.
    Returns a dictionary containing the pronunciation assessment results, or None if an error occurs.
    """
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        print("Azure Speech Service key or region not configured.")
        return {
            "error": "Azure Speech Service key or region not configured. Please check server setup."
        }

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION
    )

    # Build the pronunciation assessment config JSON
    pronunciation_config_json = {
        "referenceText": reference_text,
        "gradingSystem": "HundredMark",  # Previously speechsdk.PronunciationGradingSystem.HundredMark
        "granularity": "Phoneme",  # Previously speechsdk.PronunciationAssessmentGranularity.Phoneme
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
            result_data = json.loads(
                result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
            )
            print(f"\n{result_data}\n")
            return result_data
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


def _parse_azure_results(raw_azure_json):
    """
    Parses the raw Azure Speech Service JSON response to extract relevant scores
    for displaying in graphs.
    """
    parsed_data = {
        "overall_pronunciation_score": "N/A",
        "overall_content_score": "N/A",
        "breakdown": {
            "accuracy_score": "N/A",
            "fluency_score": "N/A",
            "prosody_score": "N/A",
            "grammar_score": "N/A",
            "vocabulary_score": "N/A",
        },
        "word_details": [],
        "raw_json": raw_azure_json,  # Include raw for debugging/full view if needed
    }

    try:
        if "NBest" in raw_azure_json and raw_azure_json["NBest"]:
            nbest = raw_azure_json["NBest"][0]  # Take the top recognition result

            if "PronunciationAssessment" in nbest:
                pa = nbest["PronunciationAssessment"]
                parsed_data["overall_pronunciation_score"] = pa.get("PronScore", "N/A")
                parsed_data["breakdown"]["accuracy_score"] = pa.get(
                    "AccuracyScore", "N/A"
                )
                parsed_data["breakdown"]["fluency_score"] = pa.get(
                    "FluencyScore", "N/A"
                )
                parsed_data["breakdown"]["prosody_score"] = pa.get(
                    "ProsodyScore", "N/A"
                )
                # Azure's PA doesn't directly provide a single "Content Score" or Grammar/Vocabulary at top level
                # These are typically derived or obtained from other analysis or specific assessment modes.
                # If these come from a different part of the PA JSON, you'd extract them here.
                # For this example, we'll try to find them from NBest or derive them.

            # Content score often comes from the combined analysis of grammar and vocabulary.
            # Azure's PA usually provides grammar and vocabulary scores within the detailed results,
            # sometimes as part of "Words" or other top-level properties if enabled.
            # Let's assume for this example, "GrammarScore" and "VocabularyScore" exist in NBest.
            # You might need to adjust this based on the exact PA output you receive.
            if "GrammarScore" in nbest:
                parsed_data["breakdown"]["grammar_score"] = nbest["GrammarScore"]
            if "VocabularyScore" in nbest:
                parsed_data["breakdown"]["vocabulary_score"] = nbest["VocabularyScore"]
            # A simple way to derive "Content Score" for demonstration if not directly provided:
            if isinstance(
                parsed_data["breakdown"]["grammar_score"], (int, float)
            ) and isinstance(
                parsed_data["breakdown"]["vocabulary_score"], (int, float)
            ):
                parsed_data["overall_content_score"] = (
                    parsed_data["breakdown"]["grammar_score"]
                    + parsed_data["breakdown"]["vocabulary_score"]
                ) / 2
                parsed_data["overall_content_score"] = round(
                    parsed_data["overall_content_score"]
                )  # Round for display

            if "Words" in nbest:
                for word in nbest["Words"]:
                    word_info = {
                        "word": word.get("Word", "N/A"),
                        "accuracy": word.get("PronunciationAssessment", {}).get(
                            "AccuracyScore", "N/A"
                        ),
                        "error_type": word.get("ErrorType", "N/A"),
                        "phonemes": [],
                    }
                    if "Phonemes" in word:
                        for phoneme in word["Phonemes"]:
                            word_info["phonemes"].append(
                                {
                                    "phoneme": phoneme.get("Phoneme", "N/A"),
                                    "accuracy": phoneme.get(
                                        "PronunciationAssessment", {}
                                    ).get("AccuracyScore", "N/A"),
                                    "error_type": phoneme.get("ErrorType", "N/A"),
                                }
                            )
                    parsed_data["word_details"].append(word_info)

    except Exception as e:
        print(f"Error parsing Azure results: {e}")
        parsed_data["error"] = f"Error parsing results: {e}"

    return parsed_data


# --- Flask Routes ---


@app.route("/", methods=["GET"])
def index():
    """Renders the main upload form page."""
    return render_template("index.html")


@app.route("/assess", methods=["POST"])
def assess():
    """
    Handles file upload and triggers pronunciation assessment in a background thread.
    Returns a job_id to the client for polling.
    """
    if "audio_file" not in request.files:
        flash("No file part", "error")
        return redirect(request.url)

    file = request.files["audio_file"]
    reference_text = request.form.get("reference_text", "").strip()

    if file.filename == "":
        flash("No selected file", "error")
        return redirect(request.url)

    if not reference_text:
        flash("Reference text is required for assessment.", "error")
        return redirect(request.url)

    if file:
        job_id = str(uuid.uuid4())
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        print(f"File saved to: {filepath} for job ID: {job_id}")

        # Initialize job status
        ASSESSMENT_RESULTS[job_id] = {"status": "pending", "data": None, "error": None}

        # Run the assessment in a separate thread
        thread = threading.Thread(
            target=_run_assessment_task, args=(job_id, filepath, reference_text)
        )
        thread.start()

        flash("File uploaded. Assessment is processing. Please wait...", "info")
        # Return job_id so frontend can poll for results
        return jsonify(
            {
                "job_id": job_id,
                "message": "Assessment started. Polling for results is recommended.",
            }
        )

    return redirect(url_for("index"))


def _run_assessment_task(job_id, filepath, reference_text):
    """
    Background task to perform audio extraction, Azure assessment, and store results.
    """
    extracted_audio_path = None
    try:
        # 1. Extract audio
        print(f"[{job_id}] Extracting audio from {filepath}...")
        extracted_audio_path = extract_audio(filepath)

        if not extracted_audio_path:
            error_msg = "Error: Could not extract audio from the file."
            print(f"[{job_id}] {error_msg}")
            ASSESSMENT_RESULTS[job_id].update({"status": "error", "error": error_msg})
            return

        # 2. Send to Azure for assessment
        print(f"[{job_id}] Sending audio to Azure for pronunciation assessment...")
        raw_assessment_results = assess_pronunciation(
            extracted_audio_path, reference_text
        )

        if raw_assessment_results and "error" not in raw_assessment_results:
            # 3. Parse and store results
            parsed_results = _parse_azure_results(raw_assessment_results)
            ASSESSMENT_RESULTS[job_id].update(
                {"status": "complete", "data": parsed_results}
            )
            print(f"[{job_id}] Assessment complete. Results stored.")
        else:
            error_msg = raw_assessment_results.get(
                "error", "Unknown error during Azure assessment."
            )
            ASSESSMENT_RESULTS[job_id].update({"status": "error", "error": error_msg})
            print(f"[{job_id}] Assessment failed: {error_msg}")

    except Exception as e:
        error_msg = f"An unexpected error occurred during background assessment: {e}"
        print(f"[{job_id}] {error_msg}")
        ASSESSMENT_RESULTS[job_id].update({"status": "error", "error": error_msg})
    finally:
        # Clean up temporary files
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"[{job_id}] Cleaned up uploaded file: {filepath}")
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            try:
                os.remove(extracted_audio_path)
                # Remove the temporary directory created by tempfile.mkdtemp()
                temp_dir_of_audio = os.path.dirname(extracted_audio_path)
                if os.path.exists(temp_dir_of_audio):
                    os.rmdir(temp_dir_of_audio)
                print(
                    f"[{job_id}] Cleaned up temporary audio and directory: {extracted_audio_path}"
                )
            except Exception as e:
                print(
                    f"[{job_id}] Error cleaning up temp audio file {extracted_audio_path}: {e}"
                )


@app.route("/get_results/<job_id>", methods=["GET"])
def get_results(job_id):
    """
    Endpoint for frontend to poll and retrieve assessment results.
    """
    result_data = ASSESSMENT_RESULTS.get(job_id)
    if not result_data:
        return (
            jsonify({"status": "not_found", "message": "Job ID not found or expired."}),
            404,
        )

    return jsonify(result_data)


# --- Main Execution ---
if __name__ == "__main__":
    # Create the uploads folder if it doesn't exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    # Run the Flask app
    app.run(
        debug=True
    )  # debug=True allows for automatic reloading and better error messages
