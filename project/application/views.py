from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.models import User

import base64, cv2, numpy as np
from django.http import JsonResponse
import mediapipe as mp

from django.views.decorators.csrf import csrf_exempt

import statistics

# Initialize Mediapipe Pose once
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Global calibration baseline (shared)
user_baseline = {"vertical": None, "forward": None}

# Create your views here.


def index(request):

    return render(request, 'index.html')

def login_page(request):

    if request.method == "POST":

        if 'login_button' in request.POST:
            username = request.POST.get('email')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('dashboard')
            else:
                messages.error(request, "Invalid email id or password")

        elif 'register_button' in request.POST:
            username = request.POST.get('email')
            first_name = request.POST.get('name')
            password = request.POST.get('password')

            if User.objects.filter(username=username).exists():
                messages.error(request, "User alraedy exists")
            else:
                user = User.objects.create_user(username=username, password=password, first_name=first_name)
                messages.success(request, "Registration successful! Please login.")

    return render(request, 'login.html')

def dashboard(request):

    return render(request, "dashboard.html")

def settings(request):
    
    return render(request, "settings.html")


@csrf_exempt
def process_frame(request):
    """
    Processes each webcam frame, compares posture metrics against
    the user's calibrated baseline, and classifies posture.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    data_url = request.POST.get("frame")
    if not data_url:
        return JsonResponse({"error": "No frame received"}, status=400)

    try:
        # Decode the base64 frame
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert BGR → RGB for Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            return JsonResponse({"posture": "unknown"})

        # Extract key landmarks
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        # Average coordinates
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        nose_x = nose.x

        # Current metrics
        vertical_ratio = (hip_y - shoulder_y)
        forward_offset = abs(nose_x - shoulder_x)

        # --- Adaptive baseline comparison ---
        if user_baseline["vertical"] is None or user_baseline["forward"] is None:
            # No calibration done yet
            posture = "unknown"
        else:
            vertical_change = user_baseline["vertical"] - vertical_ratio
            forward_change = forward_offset - user_baseline["forward"]

            # Tolerance thresholds (tune these)
            if vertical_change > 0.03 or forward_change > 0.02:
                posture = "slouching"
            else:
                posture = "upright"

        return JsonResponse({
            "posture": posture,
            "vertical_ratio": vertical_ratio,
            "forward_offset": forward_offset,
            "baseline": user_baseline
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
@csrf_exempt
def calibrate_posture(request):
    """
    Receives a short burst of upright frames to compute baseline posture.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        frames = request.POST.getlist("frames[]")  # multiple base64 frames
        verticals, forwards = [], []

        for data_url in frames:
            if not data_url:
                continue

            # Decode the frame
            header, encoded = data_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 🩶 Skip invalid or blank frames
            if frame is None or frame.size == 0:
                print("⚠️ Skipping empty frame during calibration")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if not results.pose_landmarks:
                continue

            lm = results.pose_landmarks.landmark
            ls, rs = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            lh, rh = lm[mp_pose.PoseLandmark.LEFT_HIP.value], lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            nose = lm[mp_pose.PoseLandmark.NOSE.value]

            shoulder_y = (ls.y + rs.y) / 2
            hip_y = (lh.y + rh.y) / 2
            shoulder_x = (ls.x + rs.x) / 2
            verticals.append(hip_y - shoulder_y)
            forwards.append(abs(nose.x - shoulder_x))

        if not verticals:
            return JsonResponse({"error": "No valid landmarks"}, status=400)

        user_baseline["vertical"] = statistics.mean(verticals)
        user_baseline["forward"] = statistics.mean(forwards)

        print("✅ Calibration baseline saved:", user_baseline)
        return JsonResponse({"status": "success", "baseline": user_baseline})

    except Exception as e:
        import traceback
        print("❌ Calibration error:", traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)