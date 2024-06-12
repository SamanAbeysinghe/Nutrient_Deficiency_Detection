from flask import Flask, render_template, request, jsonify, redirect, make_response
from flask.helpers import url_for
from image_classifier import ImageClassifier

app = Flask(__name__)
model_path_1 = 'Complete_m.tflite'
model_path_2 = 'verify.tflite'
class_labels_1 = [
    'Nitrogen', 'Potassium', 'Phosphorus', 'Magnesium', 'Healthy'
]
class_labels_2 = ['Black_Pepper', 'Other']

classifier_1 = ImageClassifier(model_path_2, class_labels_2)
classifier_2 = ImageClassifier(model_path_1, class_labels_1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test', methods=['GET', 'POST'])
def test():
    return render_template('varieties.html')


@app.route('/Diagnose_N', methods=['GET', 'POST'])
def Diagnose_N():
    return render_template('Diagnose_N.html')


@app.route('/Diagnose_K', methods=['GET', 'POST'])
def Diagnose_K():
    return render_template('Diagnose_K.html')


@app.route('/Diagnose_Mg', methods=['GET', 'POST'])
def Diagnose_Mg():
    return render_template('Diagnose_Mg .html')


@app.route('/Diagnose_P', methods=['GET', 'POST'])
def Diagnose_P():
    return render_template('Diagnose_P.html')


@app.route('/nitrogenPredicted', methods=['GET', 'POST'])
def nitrogenPredicted():
    return render_template('Nitrogen.html')


@app.route('/potassiumPredicted', methods=['GET', 'POST'])
def potassiumPredicted():
    return render_template('Potassium.html')


@app.route('/magnesiumPredicted', methods=['GET', 'POST'])
def magnesiumPredicted():
    return render_template('Magnesium.html')


@app.route('/phosphorusPredicted', methods=['GET', 'POST'])
def phosphorusPredicted():
    return render_template('Phosphorus.html')


@app.route('/diseasePredicted', methods=['GET', 'POST'])
def diseasePredicted():
    return render_template('Disease.html')


@app.route('/Solution_N', methods=['GET', 'POST'])
def Solution_N():
    try:
        if request.method == 'POST':
            data = request.get_json()

            question1 = data.get('question1')
            question2 = data.get('question2')
            question3 = data.get('question3')
            question4 = data.get('question4')
            question5 = data.get('question5')

            if (question1 == 'no' and question2 == 'no' and question4 == 'no'):
                return jsonify({'redirect': '/nitrogenPredicted'}), 200
            else:
                return jsonify({'redirect': '/diseasePredicted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    return make_response('Invalid request', 400)


@app.route('/Solution_K', methods=['GET', 'POST'])
def Solution_K():
    try:
        if request.method == 'POST':
            data = request.get_json()

            question1 = data.get('question1')
            question2 = data.get('question2')
            question3 = data.get('question3')
            question4 = data.get('question4')

            if (question1 == 'yes' and question2 == 'yes' and question3 == 'no'
                    and question4 == 'no'):
                return jsonify({'redirect': '/potassiumPredicted'}), 200
            elif (question2 == 'no' and question3 == 'no'
                  and question4 == 'no'):
                return jsonify({'redirect': '/potassiumPredicted'}), 200
            else:
                return jsonify({'redirect': '/diseasePredicted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    return make_response('Invalid request', 400)


@app.route('/Solution_Mg', methods=['GET', 'POST'])
def Solution_Mg():
    try:
        if request.method == 'POST':
            data = request.get_json()

            question1 = data.get('question1')
            question2 = data.get('question2')
            question3 = data.get('question3')

            if (question1 == 'yes' and question2 == 'no'
                    and question3 == 'yes'):
                return jsonify({'redirect': '/magnesiumPredicted'}), 200
            elif (question1 == 'no' and question2 == 'no'
                  and question3 == 'no'):
                return jsonify({'redirect': '/magnesiumPredicted'}), 200
            else:
                return jsonify({'redirect': '/diseasePredicted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    return make_response('Invalid request', 400)


@app.route('/Solution_P', methods=['GET', 'POST'])
def Solution_P():
    try:

        if request.method == 'POST':
            data = request.get_json()

            question1 = data.get('question1')
            question2 = data.get('question2')
            question3 = data.get('question3')
            question4 = data.get('question4')

            if (question1 == 'no' and question2 == 'no' and question3 == 'no'
                    and question4 == 'no'):
                return jsonify({'redirect': '/phosphorusPredicted'}), 200
            elif (question1 == 'no' and question4 == 'yes'):
                return jsonify({'redirect': '/phosphorusPredicted'}), 200
            else:
                return jsonify({'redirect': '/diseasePredicted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    return make_response('Invalid request', 400)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    try:

        image_file = request.files['file']
        image_bytes = image_file.read()

        prediction_1 = classifier_1.predict(image_bytes)
        if (prediction_1 == 'Black_Pepper'):
            prediction_2 = classifier_2.predict(image_bytes)
            if (prediction_2 == 'Nitrogen'):
                return redirect(url_for('Diagnose_N'))
            elif (prediction_2 == 'Potassium'):
                return redirect(url_for('Diagnose_K'))
            elif (prediction_2 == 'Phosphorus'):
                return redirect(url_for('Diagnose_P'))
            elif (prediction_2 == 'Magnesium'):
                return redirect(url_for('Diagnose_Mg'))
            else:
                prediction_2 = 'This is a healthy leaf'
        else:
            prediction_2 = "This is not a black pepper leaf"

        return jsonify({'result': prediction_2})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
