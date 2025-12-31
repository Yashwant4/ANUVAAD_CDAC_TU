# ANUVAAD_CDAC_TU
Multilingual Chat app and translation engine backend


This repository contains the deployment-ready backend for the SaralVarta Multilingual Chat Platform.

📦 System Requirements

Hardware

Minimum: 8GB RAM, 4-core CPU.

Recommended: 16GB RAM, NVIDIA GPU (8GB+ VRAM) with CUDA support.

Storage: ~5GB free space for model weights.

Software

Python 3.9+

CUDA Toolkit (if using GPU)

🛠️ Deployment Steps

Environment Setup:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Model Authorization:
Some AI4Bharat models may require you to accept their terms on Hugging Face. Ensure you are logged in via CLI if necessary:

huggingface-cli login


Running for Production:
While python api.py works for testing, use a production WSGI server like gunicorn for deployment:

pip install gunicorn
gunicorn --bind 0.0.0.0:5000 api:app --timeout 120


🧠 Models Used

Detection: AI4Bharat/IndicLID

Translation: AI4Bharat/IndicTrans2

👥 Contributors & Credits

Team: Yashwant Kumar Upadhyay, Vikrant Kumar, Medhabrata Konwar, Debashis Bhuyan, Bhargab Jyoti Bhuyan.

Guides:

Anil Kumar Gupta (CDAC, Noida)

Dr. Nabajyoti Medhi (Tezpur University)
