# akt-doc-classify
Mortgage Document Classifier

Problem Description:

The problem consists of predicting the type of the document based on hashed document data. This problem is a multi-class classification problem with 14 target variables and hashed document words.

Data Description:

The data consists of 14 document labels and hashed document content.

Explanation of Modeling Choice:

I used LSTM for document classification. I tried adding more layers to the network and the network that gave best performance on the validation dataset, is used for classification. 

Web Page Development:

Flask 
Restful API: For web service and it returns JSON formatted output
CSS : Bootstrap to develop basic webpage where it contains text box to add document text and submit button to take an predict the class.

Deployment:

Deployed web application in Azure Webservice

Below is the link for accessing web application hosted in Azure

https://akt-doc-classify.azurewebsites.net/

API docs:

http://akt-doc-classify.azurewebsites.net/api/docs/

Execution Instructions:

Type 1 Simple Program execution:

Step1: Clone the git repository.

Step2: Install requirements.txt file

Pip install -r requirements.txt

Step3: Run below command to execute the python file

Python run.py

Access the website at localhost:50000.

I also build predict API, that we can use through postman.
