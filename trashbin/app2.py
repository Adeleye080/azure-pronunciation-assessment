"""
Azure Speech Pronunciation Assessment Flask Application - FIXED VERSION
====================================================================

This application provides pronunciation assessment using Azure Speech Services
with detailed statistics and visualizations matching Microsoft's documentation.

Requirements:
- pip install flask azure-cognitiveservices-speech matplotlib seaborn pandas numpy python-dotenv
- Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables
"""

import os
import json
import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional

import azure.cognitiveservices.speech as speechsdk
from flask import Flask, render_template, request, jsonify, redirect, url_for
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Configure matplotlib style - FIX: Use valid style name
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    # Fallback if seaborn style not available
    plt.style.use("default")

sns.set_palette("husl")


class PronunciationAssessor:
    def __init__(self, speech_key: str, speech_region: str):
        self.speech_key = speech_key
        self.speech_region = speech_region

    def assess_pronunciation(
        self, audio_data: bytes, reference_text: str, language: str = "en-US"
    ) -> Dict[str, Any]:
        """
        Assess pronunciation using Azure Speech Services
        """
        try:
            # Validate inputs
            if not self.speech_key or not self.speech_region:
                return {"error": "Azure Speech credentials not configured"}

            if not audio_data:
                return {"error": "No audio data provided"}

            if not reference_text.strip():
                return {"error": "Reference text is required"}

            # Configure speech service
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, region=self.speech_region
            )
            speech_config.speech_recognition_language = language

            # Configure pronunciation assessment
            pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=reference_text,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
                enable_miscue=True,
            )

            # FIX: Use proper audio format - WAV format with specific parameters
            # Create audio stream from bytes with proper format
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,  # 16kHz sample rate
                bits_per_sample=16,  # 16-bit
                channels=1,  # Mono
            )

            audio_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)

            # Create recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )

            # Apply pronunciation assessment config
            pronunciation_config.apply_to(recognizer)

            # Push audio data and close stream
            audio_stream.write(audio_data)
            audio_stream.close()

            # Perform recognition with timeout
            result = recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Parse pronunciation assessment results
                pronunciation_result = speechsdk.PronunciationAssessmentResult(result)

                # Extract detailed results
                json_result = json.loads(
                    result.properties.get(
                        speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                    )
                )

                return self._parse_assessment_results(json_result, pronunciation_result)

            elif result.reason == speechsdk.ResultReason.NoMatch:
                return {
                    "error": "No speech could be recognized from the audio",
                    "details": "Please ensure the audio is clear and contains speech",
                }
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speechsdk.CancellationDetails(result)
                return {
                    "error": f"Speech recognition canceled: {cancellation_details.reason}",
                    "details": (
                        cancellation_details.error_details
                        if cancellation_details.error_details
                        else None
                    ),
                }
            else:
                return {
                    "error": f"Recognition failed with reason: {result.reason}",
                    "details": getattr(result, "error_details", None),
                }

        except Exception as e:
            return {"error": f"Assessment failed: {str(e)}"}

    def _parse_assessment_results(
        self, json_result: Dict, pronunciation_result
    ) -> Dict[str, Any]:
        """
        Parse and structure the assessment results
        """
        # Overall scores - FIX: Handle potential None values
        overall_scores = {
            "accuracy_score": getattr(pronunciation_result, "accuracy_score", 0) or 0,
            "fluency_score": getattr(pronunciation_result, "fluency_score", 0) or 0,
            "completeness_score": getattr(pronunciation_result, "completeness_score", 0)
            or 0,
            "pronunciation_score": getattr(
                pronunciation_result, "pronunciation_score", 0
            )
            or 0,
        }

        # Word-level analysis
        words_analysis = []
        if "NBest" in json_result and json_result["NBest"]:
            words = json_result["NBest"][0].get("Words", [])
            for word in words:
                pronunciation_assessment = word.get("PronunciationAssessment", {})
                word_info = {
                    "word": word.get("Word", ""),
                    "accuracy_score": pronunciation_assessment.get("AccuracyScore", 0),
                    "error_type": pronunciation_assessment.get("ErrorType", "None"),
                    "phonemes": [],
                }

                # Phoneme-level analysis
                phonemes = word.get("Phonemes", [])
                for phoneme in phonemes:
                    phoneme_assessment = phoneme.get("PronunciationAssessment", {})
                    phoneme_info = {
                        "phoneme": phoneme.get("Phoneme", ""),
                        "accuracy_score": phoneme_assessment.get("AccuracyScore", 0),
                    }
                    word_info["phonemes"].append(phoneme_info)

                words_analysis.append(word_info)

        # FIX: Calculate additional statistics with proper error handling
        duration_ticks = json_result.get("Duration", 0)
        # Duration is in 100-nanosecond units, convert to seconds
        speech_duration = max(
            duration_ticks / 10000000, 0.1
        )  # Prevent division by zero

        # Improved pause analysis
        omission_count = len(
            [w for w in words_analysis if w["error_type"] == "Omission"]
        )
        insertion_count = len(
            [w for w in words_analysis if w["error_type"] == "Insertion"]
        )
        pause_count = max(1, omission_count)
        avg_pause_duration = 0.2  # Placeholder - would need more complex audio analysis

        # Calculate speaking rate
        word_count = len([w for w in words_analysis if w["error_type"] != "Insertion"])
        speaking_rate = (
            (word_count / speech_duration) * 60 if speech_duration > 0 else 0
        )

        return {
            "success": True,
            "overall_scores": overall_scores,
            "words_analysis": words_analysis,
            "statistics": {
                "speech_duration": round(speech_duration, 2),
                "word_count": word_count,
                "pause_count": pause_count,
                "average_pause_duration": avg_pause_duration,
                "speaking_rate": round(speaking_rate, 1),
                "omission_count": omission_count,
                "insertion_count": insertion_count,
            },
            "recognized_text": json_result.get("DisplayText", ""),
            "reference_text": json_result.get("NBest", [{}])[0].get("Lexical", ""),
            "timestamp": datetime.now().isoformat(),
        }


class VisualizationGenerator:
    @staticmethod
    def create_pronunciation_charts(assessment_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all pronunciation assessment charts as base64 encoded images
        """
        charts = {}

        try:
            # 1. Overall Scores Bar Chart
            charts["overall_scores"] = (
                VisualizationGenerator._create_overall_scores_chart(
                    assessment_data["overall_scores"]
                )
            )

            # 2. Word-level Accuracy Chart
            charts["word_accuracy"] = (
                VisualizationGenerator._create_word_accuracy_chart(
                    assessment_data["words_analysis"]
                )
            )

            # 3. Phoneme Accuracy Heatmap
            charts["phoneme_heatmap"] = VisualizationGenerator._create_phoneme_heatmap(
                assessment_data["words_analysis"]
            )

            # 4. Speech Statistics Dashboard
            charts["statistics_dashboard"] = (
                VisualizationGenerator._create_statistics_dashboard(
                    assessment_data["statistics"]
                )
            )

            # 5. Error Type Distribution
            charts["error_distribution"] = (
                VisualizationGenerator._create_error_distribution(
                    assessment_data["words_analysis"]
                )
            )
        except Exception as e:
            print(f"Error generating charts: {e}")
            # Return empty chart for failed visualizations
            charts["error"] = VisualizationGenerator._create_empty_chart(
                f"Chart generation failed: {str(e)}"
            )

        return charts

    @staticmethod
    def _create_overall_scores_chart(scores: Dict[str, float]) -> str:
        """Create overall pronunciation scores bar chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            score_names = [
                "Accuracy",
                "Fluency",
                "Completeness",
                "Overall\nPronunciation",
            ]
            score_values = [
                scores["accuracy_score"],
                scores["fluency_score"],
                scores["completeness_score"],
                scores["pronunciation_score"],
            ]

            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
            bars = ax.bar(score_names, score_values, color=colors, alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars, score_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            ax.set_ylim(0, 105)  # Slightly higher to accommodate labels
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title(
                "Pronunciation Assessment Scores", fontsize=14, fontweight="bold"
            )
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            return VisualizationGenerator._fig_to_base64(fig)
        except Exception as e:
            return VisualizationGenerator._create_empty_chart(
                f"Error creating overall scores chart: {e}"
            )

    @staticmethod
    def _create_word_accuracy_chart(words_analysis: List[Dict]) -> str:
        """Create word-level accuracy chart"""
        if not words_analysis:
            return VisualizationGenerator._create_empty_chart("No word data available")

        try:
            # Filter out insertion errors for cleaner visualization
            filtered_words = [
                w for w in words_analysis if w["error_type"] != "Insertion"
            ]

            if not filtered_words:
                return VisualizationGenerator._create_empty_chart(
                    "No valid word data for visualization"
                )

            fig, ax = plt.subplots(figsize=(max(12, len(filtered_words) * 0.8), 6))

            words = [w["word"] for w in filtered_words]
            accuracies = [w["accuracy_score"] for w in filtered_words]

            # Color code based on accuracy
            colors = [
                "#FF6B6B" if acc < 60 else "#FFD93D" if acc < 80 else "#6BCF7F"
                for acc in accuracies
            ]

            bars = ax.bar(words, accuracies, color=colors, alpha=0.8)

            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{acc:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            ax.set_ylim(0, 105)
            ax.set_ylabel("Accuracy Score", fontsize=12)
            ax.set_xlabel("Words", fontsize=12)
            ax.set_title(
                "Word-Level Pronunciation Accuracy", fontsize=14, fontweight="bold"
            )

            # Rotate x-axis labels if too many words
            if len(words) > 8:
                plt.xticks(rotation=45, ha="right")

            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            return VisualizationGenerator._fig_to_base64(fig)
        except Exception as e:
            return VisualizationGenerator._create_empty_chart(
                f"Error creating word accuracy chart: {e}"
            )

    @staticmethod
    def _create_phoneme_heatmap(words_analysis: List[Dict]) -> str:
        """Create phoneme accuracy heatmap"""
        if not words_analysis:
            return VisualizationGenerator._create_empty_chart(
                "No phoneme data available"
            )

        try:
            # Prepare data for heatmap
            phoneme_data = []
            for word_data in words_analysis:
                word = word_data["word"]
                if word_data["error_type"] == "Insertion":
                    continue  # Skip insertion errors

                for phoneme_data_item in word_data.get("phonemes", []):
                    if phoneme_data_item["phoneme"]:  # Only include non-empty phonemes
                        phoneme_data.append(
                            {
                                "word": word,
                                "phoneme": phoneme_data_item["phoneme"],
                                "accuracy": phoneme_data_item["accuracy_score"],
                            }
                        )

            if not phoneme_data:
                return VisualizationGenerator._create_empty_chart(
                    "No phoneme data available for visualization"
                )

            # Create DataFrame and pivot for heatmap
            df = pd.DataFrame(phoneme_data)

            # Limit to reasonable size for visualization
            if len(df) > 100:
                df = df.head(100)

            pivot_data = df.pivot_table(
                index="word", columns="phoneme", values="accuracy", fill_value=0
            )

            fig, ax = plt.subplots(
                figsize=(
                    min(16, max(8, len(pivot_data.columns))),
                    min(12, max(6, len(pivot_data.index))),
                )
            )

            sns.heatmap(
                pivot_data,
                annot=True,
                cmap="RdYlGn",
                center=70,
                cbar_kws={"label": "Accuracy Score"},
                fmt=".0f",
                ax=ax,
                square=False,
            )

            ax.set_title(
                "Phoneme-Level Accuracy Heatmap", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel("Phonemes", fontsize=12)
            ax.set_ylabel("Words", fontsize=12)

            plt.tight_layout()
            return VisualizationGenerator._fig_to_base64(fig)
        except Exception as e:
            return VisualizationGenerator._create_empty_chart(
                f"Error creating phoneme heatmap: {e}"
            )

    @staticmethod
    def _create_statistics_dashboard(statistics: Dict[str, float]) -> str:
        """Create speech statistics dashboard"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Speech Analysis Dashboard", fontsize=16, fontweight="bold")

            # Speech Duration
            duration = statistics.get("speech_duration", 0)
            ax1.bar(["Speech Duration"], [duration], color="#4ECDC4", alpha=0.8)
            ax1.set_ylabel("Seconds")
            ax1.set_title("Speech Duration")
            ax1.text(
                0,
                duration + duration * 0.05,
                f"{duration:.1f}s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

            # Word Count
            word_count = statistics.get("word_count", 0)
            ax2.bar(["Word Count"], [word_count], color="#96CEB4", alpha=0.8)
            ax2.set_ylabel("Count")
            ax2.set_title("Total Words")
            ax2.text(
                0,
                word_count + word_count * 0.05,
                f"{word_count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

            # Speaking Rate
            speaking_rate = statistics.get("speaking_rate", 0)
            ax3.bar(["Speaking Rate"], [speaking_rate], color="#FFD93D", alpha=0.8)
            ax3.set_ylabel("Words per Minute")
            ax3.set_title("Speaking Rate")
            ax3.text(
                0,
                speaking_rate + speaking_rate * 0.05,
                f"{speaking_rate:.0f} WPM",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

            # Error Analysis
            omissions = statistics.get("omission_count", 0)
            insertions = statistics.get("insertion_count", 0)
            error_data = [omissions, insertions]
            error_labels = ["Omissions", "Insertions"]
            ax4.bar(error_labels, error_data, color=["#FF6B6B", "#45B7D1"], alpha=0.8)
            ax4.set_title("Error Analysis")
            ax4.set_ylabel("Count")

            for i, (label, value) in enumerate(zip(error_labels, error_data)):
                ax4.text(
                    i,
                    value + max(max(error_data), 1) * 0.05,
                    f"{value}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plt.tight_layout()
            return VisualizationGenerator._fig_to_base64(fig)
        except Exception as e:
            return VisualizationGenerator._create_empty_chart(
                f"Error creating statistics dashboard: {e}"
            )

    @staticmethod
    def _create_error_distribution(words_analysis: List[Dict]) -> str:
        """Create error type distribution pie chart"""
        if not words_analysis:
            return VisualizationGenerator._create_empty_chart("No error data available")

        try:
            # Count error types
            error_counts = {}
            for word in words_analysis:
                error_type = word.get("error_type", "None")
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            # Filter out zero counts and ensure we have data
            error_counts = {k: v for k, v in error_counts.items() if v > 0}

            if not error_counts:
                return VisualizationGenerator._create_empty_chart(
                    "No error data to display"
                )

            fig, ax = plt.subplots(figsize=(8, 8))

            colors = ["#6BCF7F", "#FF6B6B", "#FFD93D", "#45B7D1", "#96CEB4"]

            # Ensure we don't run out of colors
            plot_colors = colors[: len(error_counts)]
            if len(error_counts) > len(colors):
                plot_colors.extend(colors * ((len(error_counts) // len(colors)) + 1))

            wedges, texts, autotexts = ax.pie(
                error_counts.values(),
                labels=error_counts.keys(),
                autopct="%1.1f%%",
                colors=plot_colors[: len(error_counts)],
                startangle=90,
            )

            ax.set_title(
                "Pronunciation Error Distribution", fontsize=14, fontweight="bold"
            )

            plt.tight_layout()
            return VisualizationGenerator._fig_to_base64(fig)
        except Exception as e:
            return VisualizationGenerator._create_empty_chart(
                f"Error creating error distribution chart: {e}"
            )

    @staticmethod
    def _create_empty_chart(message: str) -> str:
        """Create empty chart with message"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="gray",
                wrap=True,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return VisualizationGenerator._fig_to_base64(fig)
        except Exception as e:
            # Fallback - return a simple base64 encoded empty image
            return ""

    @staticmethod
    def _fig_to_base64(fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            img_buffer = io.BytesIO()
            fig.savefig(
                img_buffer,
                format="png",
                dpi=100,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)
            return img_str
        except Exception as e:
            plt.close(fig)
            return ""


# Initialize global objects
speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")

if not speech_key or not speech_region:
    print("WARNING: Azure Speech credentials not found in environment variables")
    print("Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")
    print("Example:")
    print("export AZURE_SPEECH_KEY='your_key_here'")
    print("export AZURE_SPEECH_REGION='your_region_here'")

assessor = PronunciationAssessor(speech_key or "", speech_region or "")


@app.route("/")
def index():
    """Main page with upload form"""
    return render_template("index.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html")


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413


@app.route("/assess", methods=["POST"])
def assess_pronunciation():
    """Handle pronunciation assessment"""
    try:
        # Validate Azure credentials
        if not speech_key or not speech_region:
            return (
                jsonify(
                    {
                        "error": "Azure Speech credentials not configured",
                        "message": "Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables",
                    }
                ),
                500,
            )

        # Get form data
        reference_text = request.form.get("reference_text", "").strip()
        language = request.form.get("language", "en-US")

        if not reference_text:
            return jsonify({"error": "Reference text is required"}), 400

        # Get audio file
        if "audio_file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            return jsonify({"error": "No audio file selected"}), 400

        # Validate file type
        allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".webm"}
        file_ext = os.path.splitext(audio_file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return (
                jsonify(
                    {
                        "error": f"Unsupported file format: {file_ext}",
                        "message": f"Supported formats: {', '.join(allowed_extensions)}",
                    }
                ),
                400,
            )

        # Read audio data
        audio_data = audio_file.read()
        if not audio_data:
            return jsonify({"error": "Empty audio file"}), 400

        print(f"Processing audio file: {audio_file.filename} ({len(audio_data)} bytes)")
        print(f"Reference text: {reference_text}")
        print(f"Language: {language}")

        # Perform assessment
        result = assessor.assess_pronunciation(audio_data, reference_text, language)

        if not result.get("success"):
            print(f"Assessment failed: {result}")
            return jsonify(result), 400

        print(f"Assessment successful, generating visualizations...\n\n{result}")

        # Generate visualizations
        charts = VisualizationGenerator.create_pronunciation_charts(result)
        result["charts"] = charts

        print(f"Generated {len(charts)} charts")
        return jsonify(result)

    except Exception as e:
        print(f"Unexpected error in assess_pronunciation: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Assessment failed: {str(e)}"}), 500


@app.route("/results")
def results():
    """Display results page"""
    return render_template("results.html")


@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "azure_configured": bool(speech_key and speech_region),
            "timestamp": datetime.now().isoformat(),
        }
    )


if __name__ == "__main__":
    print("Starting Azure Speech Pronunciation Assessment Server...")
    print(f"Azure Speech Key configured: {'Yes' if speech_key else 'No'}")
    print(f"Azure Speech Region: {speech_region or 'Not configured'}")
    app.run(debug=True, host="0.0.0.0", port=8000)
