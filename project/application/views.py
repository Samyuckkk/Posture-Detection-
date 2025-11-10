from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Count, Q
from django.db.models.functions import TruncHour, TruncDay, TruncMonth

import base64, cv2, numpy as np, mediapipe as mp
from keras import models as keras_models
from numpy.linalg import norm
import traceback

from .models import PostureLog


# ---------- LOAD MODEL & REFERENCES ----------

MODEL = keras_models.load_model("application/posture_model.h5")
LABELS = np.load("application/labels.npy")

# Optional reference posture vectors (for fallback)
try:
    UPRIGHT_VEC = np.load("application/upright.npy")
    SLOUCH_VEC = np.load("application/slouch.npy")
    print("✅ Loaded upright.npy and slouch.npy reference vectors.")
except Exception as e:
    print("⚠️ Reference posture vectors not found:", e)
    UPRIGHT_VEC = SLOUCH_VEC = None


# ---------- MEDIAPIPE SETUP ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)


# ---------- UTILITY ----------
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-6)


# ---------- MAIN VIEWS ----------
def index(request):
    return render(request, "index.html")


def login_page(request):
    """Handles both login and registration."""
    if request.method == "POST":
        if 'login_button' in request.POST:
            username = request.POST.get("email")
            password = request.POST.get("password")
            user = authenticate(request, username=username, password=password)

            if user:
                login(request, user)
                return redirect("dashboard")
            else:
                messages.error(request, "Invalid email ID or password")

        elif 'register_button' in request.POST:
            username = request.POST.get("email")
            first_name = request.POST.get("name")
            password = request.POST.get("password")

            if User.objects.filter(username=username).exists():
                messages.error(request, "User already exists")
            else:
                User.objects.create_user(username=username, password=password, first_name=first_name)
                messages.success(request, "Registration successful! Please login.")

    return render(request, "login.html")


def dashboard(request):
    """Main dashboard view with today's slouch count."""
    if request.user.is_authenticated:
        today = timezone.now().date()
        slouch_count = PostureLog.objects.filter(
            user=request.user,
            posture__in=["slouch", "slouching"],
            timestamp__date=today
        ).count()
    else:
        slouch_count = 0

    return render(request, "dashboard.html", {"slouch_count": slouch_count})


def settings(request):
    return render(request, "settings.html")


# ---------- ML INFERENCE ----------
@csrf_exempt
def process_frame(request):
    """Processes incoming webcam frame, predicts posture, and stores logs."""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    data_url = request.POST.get("frame")
    if not data_url:
        return JsonResponse({"error": "No frame received"}, status=400)

    try:
        # --- Decode the base64 frame ---
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            print("❌ Empty frame received.")
            return JsonResponse({"error": "Empty frame"}, status=400)

        # --- Process with Mediapipe ---
        frame_flipped = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            print("⚠️ No landmarks detected.")
            return JsonResponse({"posture": "unknown"})

        # --- Extract upper-body landmarks ---
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        base_x = (left_shoulder.x + right_shoulder.x) / 2
        base_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        if shoulder_width < 1e-5:
            shoulder_width = 1.0  # avoid divide by zero

        features = []
        for lm in landmarks:
            features.append((lm.x - base_x) / shoulder_width)
            features.append((lm.y - base_y) / shoulder_width)
        features = np.array(features).reshape(1, -1)

        # --- Predict using trained model ---
        probs = MODEL.predict(features, verbose=0)[0]
        confidence = np.max(probs)
        pred_label = LABELS[np.argmax(probs)].lower()

        # --- Confidence filtering + fallback ---
        if confidence < 0.6 and UPRIGHT_VEC is not None and SLOUCH_VEC is not None:
            upright_sim = cosine_similarity(features.flatten(), UPRIGHT_VEC)
            slouch_sim = cosine_similarity(features.flatten(), SLOUCH_VEC)

            if upright_sim > slouch_sim + 0.02:
                posture = "upright"
            elif slouch_sim > upright_sim + 0.02:
                posture = "slouch"
            else:
                posture = "unknown"
        else:
            posture = pred_label
            if posture == "slouching":
                posture = "slouch"

        print(f"🧠 Predicted posture: {posture} (conf={confidence:.2f})")

        # --- Save to database ---
        try:
            PostureLog.objects.create(
                user=request.user if request.user.is_authenticated else None,
                posture=posture,
                ear_shoulder_distance=None,
            )
            print("💾 Saved posture log:", posture)
        except Exception as db_err:
            print("❌ DB save error:", db_err)

        return JsonResponse({"posture": posture})

    except Exception as e:
        print("❌ Full process_frame error:\n", traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)


# ---------- CALIBRATION ----------
@csrf_exempt
def calibrate_posture(request):
    """Simulated calibration success."""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    return JsonResponse({
        "status": "success",
        "baseline": {"ear_shoulder": 0.1234},
        "frames_used": 5
    })


# ---------- ANALYTICS ----------
@csrf_exempt
def slouch_count_today(request):
    """Return total slouches for the logged-in user today."""
    if not request.user.is_authenticated:
        return JsonResponse({"slouches_today": 0})

    today = timezone.now().date()
    count = PostureLog.objects.filter(
        user=request.user,
        posture__in=["slouch", "slouching"],
        timestamp__date=today
    ).count()
    return JsonResponse({"slouches_today": count})


@csrf_exempt
def posture_chart_data(request, period="hourly"):
    """Return posture summary aggregated hourly/daily/monthly."""
    if not request.user.is_authenticated:
        return JsonResponse({"labels": [], "data": []})

    qs = PostureLog.objects.filter(user=request.user)

    if period == "hourly":
        qs = (
            qs.annotate(period=TruncHour("timestamp"))
            .values("period")
            .annotate(slouches=Count("id", filter=Q(posture="slouch")))
            .order_by("period")
        )
        labels = [x["period"].strftime("%I %p") for x in qs]
    elif period == "daily":
        qs = (
            qs.annotate(period=TruncDay("timestamp"))
            .values("period")
            .annotate(slouches=Count("id", filter=Q(posture="slouch")))
            .order_by("period")
        )
        labels = [x["period"].strftime("%b %d") for x in qs]
    else:  # monthly
        qs = (
            qs.annotate(period=TruncMonth("timestamp"))
            .values("period")
            .annotate(slouches=Count("id", filter=Q(posture="slouch")))
            .order_by("period")
        )
        labels = [x["period"].strftime("%b %Y") for x in qs]

    data = [x["slouches"] for x in qs]
    return JsonResponse({"labels": labels, "data": data})
