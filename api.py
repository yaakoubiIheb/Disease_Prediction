from flask import Flask
from flask_restful import Api,Resource
import json
from model import predict
from flask import request,make_response
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd



app = Flask(__name__)
cors = CORS(app)
api = Api(app)



@app.route('/prediction', methods=['POST'])
def prediction():
	
	symptoms=request.get_json(force=True)
	symptoms=symptoms["symptoms"]
	return {"result": predict.predict(symptoms)[0]}



if __name__ =="__main__":
	app.run(host="localhost", port=5000, debug=False)