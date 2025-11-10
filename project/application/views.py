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

from .models import PostureLog

# Load ML posture detection model once
MODEL = keras_models.load_model("application/posture_model.h5")
LABELS = np.load("application/labels.npy")

# Initialize Mediapipe Pose once
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.3,  # more sensitive
    min_tracking_confidence=0.3
)


# ---------- MAIN VIEWS ----------
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
                messages.error(request, "User already exists")
            else:
                user = User.objects.create_user(username=username, password=password, first_name=first_name)
                messages.success(request, "Registration successful! Please login.")

    return render(request, 'login.html')


def dashboard(request):
    if request.user.is_authenticated:
        today = timezone.now().date()
        slouch_count = PostureLog.objects.filter(
            user=request.user,
            posture__in=["slouch", "slouching"],
            timestamp__date=today
        ).count()
    else:
        slouch_count = 0

    context = {
        "slouch_count": slouch_count,
    }
    return render(request, "dashboard.html", context)



def settings(request):
    return render(request, "settings.html")


# ---------- ML INFERENCE ----------
def process_frame(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    data_url = request.POST.get("frame")
    if not data_url:
        return JsonResponse({"error": "No frame received"}, status=400)

    try:
        # Decode base64 frame
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            print("❌ Empty frame")
            return JsonResponse({"error": "Empty frame"}, status=400)

        # Flip and process through Mediapipe
        frame_flipped = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            print("⚠️ No landmarks detected — returning last known posture")
            return JsonResponse({"posture": "unknown"})

        # Extract landmark features
        landmarks = results.pose_landmarks.landmark
        base_x, base_y = landmarks[24].x, landmarks[24].y
        features = []
        for lm in landmarks:
            features.append(lm.x - base_x)
            features.append(lm.y - base_y)
        features = np.array(features).reshape(1, -1)

        # Predict posture using trained model
        prediction = LABELS[np.argmax(MODEL.predict(features, verbose=0))]
        posture = prediction.lower()
        if posture == "slouching":
            posture = "slouch"

        print("🧠 Predicted posture:", posture)

        # Save posture in DB (including unauthenticated users as anonymous)
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
        import traceback
        print("❌ Full process_frame error:\n", traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)



# ---------- SIMULATED CALIBRATION ----------
@csrf_exempt
def calibrate_posture(request):
    """
    Fake calibration — just simulates progress and success.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    # Just simulate success after few frames
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
        posture__in=["slouch", "slouching"],  # ✅ Match both
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
