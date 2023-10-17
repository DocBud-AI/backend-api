from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Define a Gradio client
gradio_client = Client("https://pyakhurel-test.hf.space/--replicas/cwmlx/")

@app.route('/api/predict', methods=['POST'])
def predict_with_gradio():
    # Get data from the request
    data = request.get_json()
    if 'message' in data:
        message = data['message']
        temperature = data.get('temperature', 0)
        max_tokens = data.get('max_tokens', 0)
        top_p = data.get('top_p', 0)
        repetition_penalty = data.get('repetition_penalty', 1)

        # Call the Gradio API
        result = gradio_client.predict(
            message,
            temperature,
            max_tokens,
            top_p,
            repetition_penalty,
            api_name="/chat"
        )

        return jsonify({'result': result})
    else:
        return jsonify({'error': 'Missing message in the request'}), 400

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)