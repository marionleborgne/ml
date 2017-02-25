from flask import Flask, render_template
from flask.ext.wtf import Form
from wtforms import TextField, SubmitField
from wtforms import validators
import dill
import gzip


# vInitialize Flask App
app = Flask(__name__)


# Initialize Form Class
# This form will take in the form data on the front end and use it to predict
# using a pre-loaded model
class PredictForm(Form):
    sentence = TextField('Text to translate:',
                         validators=[validators.required(),
                                     validators.length(max=200)])
    submit = SubmitField('Submit')

# load model and load it in memory

with gzip.open('ml/my_model.dill.gz') as fin:
    model, target_names = dill.load(fin)
print "Model loaded in memory. Ready to roll!"

def convert_to_text(prediction):
  if prediction == 'ar':
    return 'Arabic'
  elif prediction == 'de':
    return 'German'
  elif prediction == 'en':
    return 'English'
  elif prediction == 'es':
    return 'Spanish'
  elif prediction == 'fr':
    return 'French'
  elif prediction == 'it':
    return 'Italian'
  elif prediction == 'nl':
    return 'Dutch'
  elif prediction == 'pl':
    return 'Polish'
  elif prediction == 'pt':
    return 'Portuguese'
  elif prediction == 'ru':
    return 'Russian'
  else:
    return '?'
  
@app.route('/', methods=['GET', 'POST'])
def translate():
    prediction, sentence = None, None
    predict_form = PredictForm(csrf_enabled=False)

    if predict_form.validate_on_submit():

        # store the submitted values
        submitted_data = predict_form.data
        print submitted_data

        # Retrieve values from form
        sentence = submitted_data['sentence']

        # Predict the class corresponding to the sentence
        predicted_class_n = model.predict([sentence])[0]

        # Get the corresponding class name and make it pretty
        prediction = target_names[predicted_class_n]
        prediction = convert_to_text(prediction)

    # Pass the predicted class name to the fron-end
    return render_template('model.html',
                           predict_form=predict_form,
                           prediction=prediction)


# Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
