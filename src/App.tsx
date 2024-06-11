import { useState } from 'react'
import './App.css'

function App() {
  const [answer, setAnswer] = useState("");
  const [answerDone, setAnswerDone] = useState(false);
  return (
    <>
      <h1>Ask Sentry</h1>
        <form onSubmit={e => {
            e.preventDefault();

            (async () => {
                setAnswer("");
                setAnswerDone(false);
                const formData = new FormData(e.currentTarget);
                const resp = await fetch("/api/v1/questions?q="+encodeURIComponent(formData.get("question") as string) +
                    "&model="+encodeURIComponent(formData.get("model") as string));
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
                setAnswerDone(true);
            })()

        }}>
            <input name="question" placeholder="Ask a question"/>
            <select name="model">
                <option value="gpt3">GPT 3</option>
                <option value="gpt4">GPT 4</option>
                <option value="claude">Anthropic Claude</option>
                <option value="cohere">Cohere</option>
                <option value="huggingface">Huggingface</option>
            </select>
            <button type="submit">Submit</button>
        </form>
        <div className="whitespace-pre">{answer}</div>
        {answerDone ? <div className="flex gap-2">
            <button>👍</button>
            <button>👎</button>
        </div> : null}
    </>
  )
}

export default App
