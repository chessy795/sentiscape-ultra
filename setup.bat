@echo off
echo ====================================
echo  Sentiment Analysis Ultra - Setup
echo ====================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Downloading NLTK data...
python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('sentiwordnet', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); print('NLTK data ready')"
echo.
echo Setup complete! Run:
echo   python sentiscape_ultra.py
echo   python sentiscape_ultra.py data.csv text_col
echo.
pause
