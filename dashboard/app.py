import streamlit as st
import requests
import sys
from PIL import Image
import subprocess
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Forest Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔥 Forest Fire & Smoke Detection")

st.write(
    "Upload an image and the YOLO model will detect fire or smoke."
)

st.divider()


# ============================================================
# IMAGE DETECTION
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(
        image,
        use_container_width=True
    )

    st.divider()

    # Detect button
    if st.button(
        "🔍 Detect Fire / Smoke",
        type="primary"
    ):

        # Reset file position
        uploaded_file.seek(0)

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        try:

            # Send image to FastAPI
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                st.subheader("Detection Result")

                if result["detected"]:

                    st.error(
                        "🚨 FIRE / SMOKE DETECTED!"
                    )

                    for detection in result["detections"]:

                        st.write(
                            f"**{detection['class']}** — "
                            f"Confidence: "
                            f"{detection['confidence']:.2f}"
                        )

                else:

                    st.success(
                        "✅ No fire or smoke detected."
                    )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to FastAPI. "
                "Make sure the FastAPI server is running."
            )


# ============================================================
# VIDEO DETECTION
# ============================================================

st.divider()

st.header("🎥 Video Detection")

st.write(
    "Upload a video and run temporal fire/smoke verification."
)


uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"],
    key="video_uploader"
)


if uploaded_video is not None:

    # --------------------------------------------------------
    # Save uploaded video temporarily
    # --------------------------------------------------------

    input_video_path = "data/videos/uploaded_video.mp4"

    with open(input_video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())


    # --------------------------------------------------------
    # Display uploaded video
    # --------------------------------------------------------

    st.subheader("Uploaded Video")

    st.video(input_video_path)

    st.divider()


    # --------------------------------------------------------
    # Temporal Verification Button
    # --------------------------------------------------------

    if st.button(
        "🎬 Run Temporal Verification",
        type="primary",
        key="video_detect_button"
    ):

        with st.spinner(
            "Processing video... This may take some time."
        ):

            try:

                # Run temporal detection script
                result = subprocess.run(
                    [
                        sys.executable,
                        "src/inference/video_temporal.py",
                        input_video_path
                    ],
                    capture_output=True,
                    text=True
                )


                # ------------------------------------------------
                # Display script output
                # ------------------------------------------------

                if result.stdout:

                    st.text_area(
                        "Temporal Verification Output",
                        result.stdout,
                        height=300
                    )


                # ------------------------------------------------
                # Display errors/warnings from script
                # ------------------------------------------------

                if result.stderr:

                    st.error(
                        "Temporal Verification Error"
                    )

                    st.code(
                        result.stderr
                    )


                # ------------------------------------------------
                # Check whether script completed successfully
                # ------------------------------------------------

                if result.returncode == 0:

                    st.success(
                        "✅ Temporal verification completed!"
                    )


                    # Output video generated by the script
                    output_video = (
                        "runs/temporal_detection_h264.mp4"
                    )


                    # ------------------------------------------------
                    # Check output video
                    # ------------------------------------------------

                    if os.path.exists(output_video):

                        st.subheader(
                            "🎯 Temporal Verification Result"
                        )

                        st.video(
                            output_video
                        )

                        st.success(
                            "🔊 Audio alert has been added "
                            "to the result video."
                        )

                    else:

                        st.error(
                            "❌ Output video was not found."
                        )


                else:

                    st.error(
                        "❌ Temporal verification failed."
                    )

                    if result.stderr:

                        st.code(
                            result.stderr
                        )


            except Exception as e:

                st.error(
                    f"❌ Error running temporal verification: {e}"
                )