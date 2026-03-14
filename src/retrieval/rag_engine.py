import os
import logging
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, retriever): 
        logger.info("initializing rag engine")
        self.llm = self._setup_llm()
        self.llm_name = os.getenv("OPENAI_MODEL")
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
        logger.info("creating openai client")
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # def generate_context(self, query):
    #     result = self.retriever.search(query)
    #     context = [dict(data) for data in result]
    #     combined_prompt = []

    #     for entry in context:
    #         context_str = entry["payload"]["context"]
    #         combined_prompt.append(context_str)

    def generate_context(self, query):
        logger.info("generating retrieval context for query_length=%d", len(query))
        result = self.retriever.search(query)
        context = []
        for entry in result:
            logger.info("retrieval entry=%s", entry)
            point_id = entry.id  # the numeric ID from Qdrant
            payload = entry.payload or {}
            chunk_text = payload.get("text") or payload.get("context")

            if not chunk_text:
                logger.warning(
                    "skipping point with missing text/context point_id=%s payload_keys=%s",
                    point_id,
                    list(payload.keys()),
                )
                continue

            logger.debug("retrieved point_id=%s chunk_length=%d", point_id, len(chunk_text))
            context.append(chunk_text)

        if not context:
            logger.error("no valid context chunks found in retrieval results")
            raise ValueError("No valid retrieval context found for this query.")

        combined_prompt = "\n\n---\n\n".join(context)
        logger.info("generated combined context chunk_count=%d length=%d", len(context), len(combined_prompt))
        return combined_prompt

    
    def stream_and_store(self, stream):
        logger.info("streaming llm response")
        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content
                logger.debug("received llm delta size=%d", len(delta.content))
                yield delta.content   # for real streaming

        self.conversation_history.append({
            "role": "assistant",
            "content": full_text
        })
        logger.info("stored assistant response length=%d history_size=%d", len(full_text), len(self.conversation_history))



    def query(self, query, difficulty):
        """
        Handles conversation flow:
        - If no active question → generate an open-ended question.
        - If there is an active question → evaluate or continue the discussion.
        """
        logger.info("rag query start difficulty=%s query_length=%d", difficulty, len(query))
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
        logger.info("rag query request dispatched model=%s", self.llm_name)
        return self.stream_and_store(response)
