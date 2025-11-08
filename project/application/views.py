from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.models import User

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