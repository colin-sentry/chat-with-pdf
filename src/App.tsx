import { useState } from 'react'
import './App.css'

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  return (
    <>
      <h1>AI RAG app</h1>
        <form onSubmit={e => {
            e.preventDefault();

            (async () => {
                setAnswer("");
                const resp = await fetch("/api/v1/questions?q="+encodeURIComponent(question));
                if(!resp.ok || !resp.body) {
                    throw new Error("Bad response")
                }
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();

                let done = false;
                while(!done) {
                    let value: Uint8Array|undefined;
                    ({value, done} = await reader.read());
                    if(value) {
                        const decoded = decoder.decode(value, {stream: true});
                        setAnswer(answer => answer+decoded);
                    }
                }
            })()

        }}>
            <input value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question"/>
            <button type="submit">Submit</button>
        </form>
        <div>{answer}</div>
    </>
  )
}

export default App
