rm -f assignment4.zip
zip -r assignment4.zip . -x "*lib/datasets*" "*.ipynb_checkpoints*" "*.idea*" "*collectSubmission.sh" "*requirements.txt" "*.pyc"
