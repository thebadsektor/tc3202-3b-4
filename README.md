TC-3202 BotaniSnap-AI
Capstone Project - AI-Powered Plant Identifier and Disease Detector

Table of Contents
Introduction

Project Overview

Objectives

Features

Technologies Used

Setup and Installation

Usage Instructions

Project Structure

Contributors

Project Timeline

Changelog

Acknowledgments

License

Introduction
BotaniSnap-AI is a smart, AI-powered plant identification and disease detection web application. It aims to help gardeners, farmers, and plant enthusiasts accurately identify plants and diagnose diseases simply by uploading an image.

Project Overview
The project integrates computer vision, machine learning, and a user-friendly frontend to support plant care and diagnosis.

Background: Misidentification of plants or plant diseases can lead to ineffective care or crop loss. This project addresses that by leveraging AI for real-time recognition.

Target Audience: Plant lovers, agriculture students, farmers, hobbyists, and educators.

Real-World Application: Promotes sustainable agriculture, helps prevent disease spread, and serves as an educational tool.

Objectives
Develop a web-based tool for plant species identification and disease detection.

Integrate Roboflow and Gemini API for accurate predictions and plant care tips.

Validate the accuracy of the model with real-world plant datasets.

Features
Plant Identification: Upload an image to get the species name.

Disease Detection: Detect common plant diseases from leaf images.

Care Tips Generator: Auto-generated care tips based on plant type using Gemini API.

User-Friendly Interface: Simple and responsive web UI for ease of use.

Technologies Used
Programming Languages: Python, JavaScript

Frameworks/Libraries: React, Flask, TensorFlow

APIs: Roboflow, Gemini 2.0 Flash (for care tips)

Databases: (None yet or mention Firebase/MySQL if used)

Other Tools: Git, GitHub, Postman

Setup and Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/thebadsektor/tc3202-3b-4.git
cd tc3202-3b-4
2. Backend Setup (Python + Flask)
bash
Copy
Edit
cd backend
pip install -r requirements.txt
3. Frontend Setup (React)
bash
Copy
Edit
cd frontend
npm install
4. Configure Environment Variables
Create .env files in both backend/ and frontend/ directories. Include API keys for Roboflow and Gemini (sample below):

backend/.env

ini
Copy
Edit
ROBOFLOW_API_KEY=your_api_key
GEMINI_API_KEY=your_api_key
5. Run the Project
Run Backend
bash
Copy
Edit
cd backend
python app.py
Run Frontend
bash
Copy
Edit
cd frontend
npm start
Usage Instructions
Open your browser and navigate to http://localhost:3000

Upload a clear photo of a plant or leaf

Wait for AI results to show species and possible diseases

Read generated care tips provided on the results page

You can test API calls using Postman or cURL for backend endpoints (if applicable).

Project Structure
java
Copy
Edit
.
├── backend/
│   ├── venv
│   ├── model/
│   │   ├── plant_identification_model.h5
│   ├── routes/
│   └── app.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── utils/
│   ├── public/
│   ├── node modeules/
│   └── package.json
├── tests/
└── README.md

Contributors
[Name 1]: Lead Developer, Backend Developer

[Name 2]: Frontend Developer, UI/UX Designer

Gerald Villaran: Course Instructor

Project Timeline
Week 1-2: Research and project planning

Week 3-5: Design and setup

Week 6-10: Implementation

Week 11-12: Testing and debugging

Week 13-14: Final presentation and documentation

Changelog
[Version 1.0.0] - 2024-09-07

Initial release of the project

Added basic functionality for plant detection, disease classification, and care tips

[Version 1.1.0] - 2024-09-14

Improved user interface for plant detection

Fixed bugs related to disease classification

Updated project documentation with setup instructions

[Version 1.2.0] - 2024-09-21

Added new functionality for offline image preview

Refactored codebase for better performance

Added unit tests for disease and care tip generation modules

Acknowledgments
This project was inspired by the original repository tc3202-3b-4.
Special thanks to the developers behind the Roboflow API and Google’s Gemini model.
Gratitude to our mentor Gerald Villaran for his guidance.

License
This project adopts the same license as the original repository. See the LICENSE file for details.