git clone https://github.com/hongki/midterm_rag.git

cd midterm_rag

python3.11 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

python preprocess_pdfs.py

python main_rag.py
