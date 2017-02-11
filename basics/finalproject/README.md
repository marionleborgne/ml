# Language detector exercise

Hosted on [Heroku](https://language-detector.herokuapp.com/).

Uses [custom buildbpack](https://github.com/thenovices/heroku-buildpack-scipy) for scipy dependency.

This is a simple Flask app that will predict the language of a sentence using sci-kit Learn.

## Steps

Open the ml folder. It contains 2 files and 1 folder:
- language_detector_test.py
- language_detector.py
- paragraphs

1) Inspect the content of the paragraphs folder. That's our starting data
You'll have to figure out a way to extract features from the text

2) try running "python language_detector_test.py" and see that it's not working: no saved model is present. You will need to build a model and save it.

3) Open the "language_detector.py" file. This is where most of the work will be. Complete each task in sequence untill you get a satisfactory value for the test score.

4) Once you have a saved model, run again "python language_detector_test.py" and see that it detects the language of the sentences. You can also try to give your own sentence by running "python language_detector_test.py 'insert here whatever sentence you want' "

5) It's now time to run our server locally. Run "python controller.py", it will load the model and start a web-server.

6) Visit http://127.0.0.1:5000/ with your browser and test that you can submit a sentence in any language. If your model is well trained it should tell you the language of the sentence

7) Explore the "controller.py" file. Can you figure out what it does?

8) Explore the rest of the code. Can you figure out what the other files do?