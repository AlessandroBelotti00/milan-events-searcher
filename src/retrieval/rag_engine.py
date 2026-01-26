import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

class RAG:
    def __init__(self, retriever): 

        self.llm = self._setup_llm()
        self.llm_name = os.getenv("OPENAI_DEPLOYMENT_NAME")
        self.retriever = retriever

        self.conversation_history = []

        self.last_question = None

        self.qa_prompt_tmpl_str = """
            Le informazioni di contesto sono riportate di seguito.
            <context/user_query>:
            ---------------------
            {context}
            ---------------------

            Sulla base delle informazioni contenute in `{context}`, genera una **ricetta completa e chiara** che risponda alla richiesta dell’utente. Segui queste regole in modo preciso:

            1. **Fonte delle informazioni:**  
            - Usa **solo** le informazioni presenti in `{context}`.  
            - Non inventare ingredienti o passaggi di preparazione non presenti nel contesto.

            2. **Obiettivo della ricetta:**  
            - Fornisci una **ricetta finale e utilizzabile** per il cibo o la preparazione richiesta.  
            - La ricetta deve essere comprensibile e pratica per chi cucina a casa.

            3. **Sezione Ingredienti:**  
            - Includi una sezione intitolata **Ingredienti**.  
            - Elenca gli ingredienti usando un **elenco puntato** (`- ingrediente`).  
            - Includi le quantità se presenti nel contesto.

            4. **Sezione Preparazione:**  
            - Includi una sezione intitolata **Preparazione**.  
            - Descrivi i passaggi in **ordine chiaro e logico**.  
            - Usa paragrafi brevi o passaggi numerati se opportuno.

            5. **Livello di dettaglio:**  
            - Sii preciso e conciso, includendo tutti i passaggi essenziali.  
            - Considera un livello di abilità **intermedio per cucina casalinga**.

            6. **Tono:**  
            - Chiaro, amichevole e istruttivo.  
            - Evita storie o opinioni personali.

            7. **Formato di output:**  

            **Titolo della ricetta**

            **Ingredienti**
            - ingrediente 1  
            - ingrediente 2  
            - ingrediente 3  

            **Preparazione**
            1. Passaggio uno  
            2. Passaggio due  
            3. Passaggio tre  

            8. **Restrizioni:**  
            - Non porre domande all’utente.  
            - Non includere spiegazioni fuori dalla ricetta.  
            - Non racchiudere l’output in blocchi di codice o triple backtick.

            9. **Lingua:**  
            - Solo Italiano.

            ---------------------
            Richiesta utente: {query}
            ---------------------
            Risposta:

        """

    def _setup_llm(self):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_context(self, query):
        result = self.retriever.search(query)
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            context_str = entry["payload"]["context"]
            combined_prompt.append(context_str)

        return "\n\n---\n\n".join(combined_prompt)
    
    def stream_and_store(self, stream):
        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content
                yield delta.content   # for real streaming

        self.conversation_history.append({
            "role": "assistant",
            "content": full_text
        })



    def query(self, query, difficulty):
        """
        Handles conversation flow:
        - If no active question → generate an open-ended question.
        - If there is an active question → evaluate or continue the discussion.
        """

        context = self.generate_context(query)
        prompt = self.qa_prompt_tmpl_str.format(context=context, difficulty=difficulty, query=query)

        messages = [
            {"role": "system", "content": "You are a university examiner."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            stream=True,
        )
        
        return self.stream_and_store(response)
