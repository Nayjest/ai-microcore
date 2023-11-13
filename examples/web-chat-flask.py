"""
Requirements: pip install flask
"""
from flask import Flask, request, jsonify

import microcore as mc

mc.use_logging()
app = Flask(__name__)


@app.route("/")
def index():
    return """
    <title>Chat</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <div class="vh-100 d-flex flex-column p-3">
        <h4 class="mb-3">Chat with AI</h4>
        <div id="h" class="flex-grow-1 overflow-auto mb-3 p-2" style="border:1px solid #dee2e6;"></div>
        <div class="input-group">
            <input id="m" class="form-control" placeholder="Enter message" onkeypress="if(event.keyCode==13)send()">
            <button class="btn btn-primary" type="button" id="button-addon2" onclick=send()>Send</button>
        </div>
    </div>
    <script>
        async function send(){
            let i=document.getElementById('m'),h=document.getElementById('h'),v=i.value,m=document.createElement('div');
            m.innerHTML='<b>You:</b> '+v;
            h.append(m);
            i.value='';
            let r=await fetch('/ask',{
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({message:v})
            }).then(res=>res.json());
            m=document.createElement('div');
            m.innerHTML='<b>AI:</b> '+r.response;
            h.append(m);
            h.scrollTop=h.scrollHeight;
        }
    </script>
"""


@app.route("/ask", methods=["POST"])
def ask_llm():
    user_msg = request.json.get("message")
    response = mc.llm(user_msg)
    return jsonify({"response": response})


app.run(debug=True)
