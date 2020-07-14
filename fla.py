import flask
from flask import Flask, render_template, request, jsonify, redirect, flash
from sklearn.externals import joblib
from ngrams import ngrams_per_line, preprocess_text
from scipy.sparse import hstack, csr_matrix, vstack
import os
import json
# For time counter
from time import perf_counter

from tf_idf import readData

UPLOAD_FOLDER = './patterns/'
ALLOWED_EXTENSIONS = {'txt'}
PATTERNS_PATH = './patterns/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_input(request, modelType):
	# check if the post request has the file part
	print(request)
	if 'file' not in request.files:
		print('No file part')
		message='No file part!'
		return (False, message, '')

	file = request.files['file']
	print('File name: ', file.filename)
	newFileName = request.form['representative']
	print('NEW file name: ', newFileName)
	# if user does not select file, browser also
	# submit an empty part without filename
	if file.filename == '':
		print('No selected file')
		message='No selected file!'
		return (False, message, newFileName)
	if file and allowed_file(file.filename):
		# filename = secure_filename(file.filename)
		file.save(os.path.join(UPLOAD_FOLDER+modelType+'/'+newFileName+'_patterns.txt'))
		message='Saved file with name: '+newFileName+'.txt. Model is trained!'
		return (True, message, newFileName)
	return (False, 'ERROR', '')
	# return file
		
def handleRetrain(strings, repName, modelType):
	# parse strings to list
	print(strings)
	print(strings[0])
	strings = strings[:-1]
	stringList = strings.split(';')
	print(stringList)

	# add strings to repFile and delete from other file
	otherFilePath = PATTERNS_PATH+modelType+'/other_patterns.txt'
	repFilePath = PATTERNS_PATH+modelType+'/'+repName+'_patterns.txt'
	with open(otherFilePath, 'r', encoding='utf-8') as otherFile:
		lines = otherFile.readlines()

	with open(otherFilePath, 'w', encoding='utf-8') as otherFile:
		with open(repFilePath, 'a', encoding='utf-8') as repFile:
			for line in lines:
				if line[:-1] not in stringList:
					otherFile.write(line)
				else:
					repFile.write(line)

def resultFilter(resultDict, newRep):
	filteredDict = resultDict
	trainList = []
	if newRep in resultDict['train']:
		for string in resultDict['train'][newRep].split(';'):
			if string != '':
				tempDict = {}
				tempDict['string'] = string
				tempDict['rep'] = newRep
				trainList.append(tempDict)
	testList = []
	if newRep in resultDict['test'][newRep]:
		for string in resultDict['test'][newRep].split(';'):
			if string != '':
				tempDict = {}
				tempDict['string'] = string
				tempDict['rep'] = newRep
				testList.append(tempDict)

	filteredDict['train'] = trainList
	filteredDict['test'] = testList
	print('*'*20, '\nfiltered Dict:', filteredDict)
	filteredJson = json.dumps(filteredDict)
	print('*'*20, '\nfiltered Json:', filteredJson)
	return filteredDict

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
   return render_template('index.html')

@app.route('/table')
def table():
   return render_template('table.php')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
	if request.method == 'GET':
		return render_template('train.html')
	if request.method == 'POST':
		t1_start = perf_counter()
		modelType = request.form['modelType']
		action = request.form['action']
		if action == 'train':
			isFileOK, message, repName = handle_input(request, modelType)
			if not isFileOK:
				return render_template('train.html', message=message)
		if action == 'retrain':
			strings = request.form['strings']
			print("***STRINGGGGG***", strings)
			repName = request.form['repName']
			print("Tên cơ quan muốn kiểm tra: ", repName)
			if strings != '':
				print("HANDLERETRAIN!!!")
				handleRetrain(strings, repName, modelType)
			else:
				print("***NOT HANDLE RETRAIN***")

		# if everything is ok, train the model
		resultDict = readData(modelType)
		resultJson = resultFilter(resultDict, repName)
		# print(resultDict)
		# load new model
		global model
		model = joblib.load('pythonModels/'+modelType+'_svm.pkl')
		global tfidf
		tfidf = joblib.load('pythonModels/'+modelType+'_tfidf.pkl')
		global encode_rev
		encode_rev = joblib.load('pythonModels/'+modelType+'_encode_rev.pkl')
		# render_template('train.html', message=message, json=resultJson)
		t1_stop = perf_counter()
		print('Time Executed: ', t1_stop - t1_start)
		return resultJson
		# return render_template('train.html', message=message, json=resultJson)

@app.route('/predict', methods=['GET', 'POST'])
def make_prediction():
	if request.method == 'GET':
		return render_template('index.html')
	if request.method == 'POST':
		text = request.form['text']
		# json = request.get_json()
		print('text: ', text)
		# preprocessing text
		sen = preprocess_text(text)
		sen_tfidf = tfidf.transform([sen])
		feat = hstack([sen_tfidf]).tocsr()
		label = model.predict(feat)
		# prob = round(label[0][0], 4)
		label = encode_rev[label[0]]
		# print('json: ', json)
		# return render_template('index.html', label=label)
		return jsonify({'prediction': str(label)})

if __name__ == '__main__':
	model = joblib.load('pythonModels/scopus_svm.pkl')
	tfidf = joblib.load('pythonModels/scopus_tfidf.pkl')
	encode_rev = joblib.load('pythonModels/scopus_encode_rev.pkl')
	app.run(host='0.0.0.0', port=8000, debug=True)