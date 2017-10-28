
import pickle 
import pandas as pd 
import numpy as np 
import flask

key_ing = ['flour','egg','salt','oil','water','cilantro','garlic','milk'
           ,'pepper','tomato','corn','onion','butter','sugar','almond'
           ,'broth','chicken','beef','pork','sausage','rice','peanut'
           ,'cream','yeast','olives','lettuce','carrot','shrimp','walnut'
           ,'lemon','orange','ginger','allspice','turkey','cinnamon'
           ,'mint','parsley','thym','spinach','chive','dill','basil'
           ,'tarragon','coriander','parmesan','rosemary','lime','pecan'
           ,'peas','apple','vinegar','celery','cumin','turmeric','lamb'
           ,'cardamom','oregano','chili','cabbage','soy_sauce','mustard'
           ,'coconut_milk','raisins','nutmeg','bread','apricot','syrup'
           ,'cheddar','mozzarella','parmesan','romano','ricotta','jack'
           ,'squash','paprika','chocolate','potato','cocoa','sour_cream'
           ,'catfish','salmon','yogurt','sesame_seeds','vanilla'
           ,'feta_cheese']


def standardize_ingredient(ingredients_string):
    """ This function returns a list of ingredients 
    with standardized names. """
    ingredients_string = ','.join(ingredients_string).replace(' ','_').lower().strip()
    for i in ingredients_string.split(','):
        for j in key_ing:
            if j in i:
                ingredients_string = ingredients_string.replace(i,j)
            else: 
                pass
    ingredients = [ingredients_string.replace(',',' ')]
    return ingredients 


with open('../Data/countvectorizer.pickle', 'rb') as f:
    vec = pickle.load(f)


def count_vec(ingredients):
    ingredients = vec.transform(ingredients)
    return ingredients


with open('../Data/bestestimator.pickle', 'rb') as f1:
    estimator = pickle.load(f1)


# def predict_with_nb(x):
#     return PREDICTOR.predict(x)

app = flask.Flask(__name__)


@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, index.html
    """
    with open("cuisine.html", 'r') as viz_file:
        return viz_file.read()


@app.route("/predictcuisine", methods=["POST"])
def predict():
    data = flask.request.json
    x = data["example"]
    x_std = standardize_ingredient(x)
    ingredients = count_vec(x_std)
    cuisines = estimator.predict(ingredients)
    print(cuisines)
    # Put the result in a nice dict so we can send it as json
    results = {"cuisines": cuisines[0]}
    print(results)
    return flask.jsonify(results)


app.run(host='0.0.0.0')
app.run(debug=True)


