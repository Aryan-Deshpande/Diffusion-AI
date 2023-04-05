import flask
from flask import request, json

@app.route('/', method=["POST"])
def somefunction():
    image = request.json.get('image')
    result = x()

    return jsonify({})
