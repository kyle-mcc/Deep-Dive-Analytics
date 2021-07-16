FROM amazon/aws-lambda-python:3.8

COPY lambda_function.py requirements.txt ./

RUN pip install -r requirements.txt

CMD ["lambda_function.lambda_handler"]