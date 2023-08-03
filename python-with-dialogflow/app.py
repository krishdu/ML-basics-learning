from typing import Dict
from flask import Flask, request
from dialogflow_fulfillment import WebhookClient, QuickReplies

app = Flask(__name__)

def happyFollowUp(agent: WebhookClient) -> None:
    agent.add('Great to know you are happy.')

def sadFollowUp(agent: WebhookClient) -> None:
    agent.add('Oh, Sorry to hear your day is not good. How can I make you smile?')
    
def welcome(agent: WebhookClient) -> None:
    agent.add('How are you feeling today?')
    agent.add(QuickReplies(quick_replies=['Happy', 'Sad']))
    
def handler(agent: WebhookClient) -> None:
    intentMap = {}
    intentMap['Default Welcome Intent'] = welcome
    intentMap['Default Welcome Intent- happy'] = happyFollowUp
    intentMap['Default Welcome Intent - sad'] = sadFollowUp

    agent.handle_request(intentMap)

@app.route('/', methods = ['GET'])
def welcome_to_personify():
    return 'Welcome to Personify :)'

@app.route('/', methods=['POST'])
def webhook() -> Dict:
    """Handle webhook requests from Dialogflow."""
    # Get WebhookRequest object
    request_ = request.get_json(force=True)

    # Log request headers and body
    print(f'Request headers: {dict(request.headers)}')
    print(f'Request body: {request_}')

    # Handle request
    agent = WebhookClient(request_) 
    agent.handle_request(handler)

    # Log WebhookResponse object
    print(f'Response body: {agent.response}')

    return agent.response

# driver function
if __name__ == '__main__':
    app.run(debug = True)