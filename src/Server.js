const express = require('express')
const cors = require('cors')
const bodyparser = require('body-parser')

require('dotenv').config()

const app = express()

app.use(
    cors({'origin':'*'}),
    bodyparser.json(),
    bodyparser.urlencoded({extended: true})
)

// host react app
app.post('/chat', (req,res)=> {
    console.log(req.body)
    
    // call the model here

    res.send({generated: 'Hello World!'})
})
// host model apis

app.listen(process.env.PORT, (err)=> {
    console.log(`Server is Listening`);
})