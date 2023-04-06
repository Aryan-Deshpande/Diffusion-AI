import './App.css';
import { useState } from 'react';

function Chatcomp() {

    const [textprompt, setTextprompt] = useState('')
    const [generated, setGenerated] = useState('')
    const [messages, setMessages] = useState([{ text: 'hello there', isUser: false }])
  
    const textInference = async function(){
      console.log(textprompt)
  
      // fetch request to the backend
      await fetch('http://localhost:8080/chat', {
        headers:{
          'Content-Type': 'application/json'
        },
        method: 'POST',
        body: JSON.stringify({textprompt: textprompt})
      })
      .then(response => response.json())
      .then(data => {
        console.log(data)
        // Add generated text to messages
        setGenerated(data.generated)
        setMessages(prevMessages => [...prevMessages, { text: data.generated, isUser: false }])
      })
      .catch(err => console.log(err))
      .finally(() => console.log('Done!'))
    }
  
    const imageRendering = async function(){
      console.log('rendering image')
      // display the image, in the image container using the generated text
    }
  
    if(generated !== ''){
      imageRendering()
    }
  
    return (
      <div className="chat-box">
  
        <div className="message-container">
          {
            messages.map((message, index) => {
              return (
                <div key={index} className={message.isUser ? "user-message message" : "bot-message message"}>
                  <p>{message.text}</p>
                </div>
              )
            })
          }
        </div>
  
        <div className="input-container">
          <input type="text" placeholder='Place your prompt here' onChange={event => setTextprompt(event.target.value)}></input>
          <input type="submit" value="Submit" onClick={textInference} className="btn"></input>
        </div>
  
        <div className='image-container'>
          {
            generated !== '' ? <image src={generated} alt="Generated Image"></image> : <p>Image will be displayed here</p>
          }
        </div>
  
      </div>
    );
  }
  
  export default Chatcomp;

