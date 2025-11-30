import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

class RAG:
    def __init__(self, retriever):  
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in simple terms"},
            ]
        )

        print(response.choices[0].message["content"])

        if LOCAL_SETTINGS:
            self.llm_name = 'llama3.2:latest'
        else:
            self.llm_name = 'gpt-5'

        self.llm = self._setup_llm()
        self.retriever = retriever

        self.conversation_history = []

        self.last_question = None

        self.qa_prompt_tmpl_str = """
            Context information is below.
                <context/user_query>:
                ---------------------
                {context}
                ---------------------

                Based on the context information above, generate **one single open-ended question** related to the query below. Follow these rules precisely:

                1. **Number of questions:** 1  
                2. **Question type:** Open-ended (short essay or problem-solving)  
                3. **Difficulty level:** the the difficulty level and deepness of the question has to be {difficulty}
                4. **Content level:** University level (intermediate to advanced), aligned with the topic in `{context}`.  
                5. **LaTeX formatting:**  
                - Use inline math between single dollar signs `$...$`  
                - Use block math between double dollar signs `$$...$$`  
                - Do not escape backslashes
                6. **Tone:** Professional and educational
                7. **Output format:**  

                    ```
                    **Open-ended Question**

                    [your question here]
                    ```

                8. **Interaction flow:**  
                - Step 1: Generate and display the question only.  
                - Step 2: Wait for the user to provide their answer.  
                - Step 3: After receiving the user’s answer, evaluate it:
                    - Positive feedback if correct or partially correct, plus explanation.
                    - Constructive feedback and correct answer if incorrect.

                9. **Language:** English only.

                ---------------------
                Query: {query}
                ---------------------
                Answer:
        """

        self.evaluation_prompt = """
        Evaluate the following user answer based on the question asked previously.

        Question:
        {question}

        User Answer:
        {user_answer}
        
        Your task:
            1. Assess the answer for relevance, accuracy, completeness, and clarity.
            2. Assign a **numerical grade from 0 to 100**. Round to the nearest whole number.
            3. Provide constructive feedback and explain **exactly why you assigned this grade**.
            4. Use the following strict output format:
                - Check if the answer is relevant and correct regarding the question.
                - Provide constructive feedback.
                - If correct or partially correct → positive feedback + short explanation.
                - If incorrect → constructive feedback + correct answer.
                - Answer in English.
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

    def query(self, query, difficulty):
        """
        Handles conversation flow:
        - If no active question → generate an open-ended question.
        - If there is an active question → evaluate or continue the discussion.
        """

        # If there's an ongoing conversation (user is responding or retrying)
        if self.last_question:
            # Append new user message to conversation history
            self.conversation_history.append({"role": "user", "content": query})

            # Build evaluation prompt dynamically from conversation
            evaluation_prompt = self.evaluation_prompt.format(
                question=self.last_question,
                user_answer=query
            )

            # Add evaluator system prompt
            messages = [{"role": "system", "content": "You are a university evaluator."}]
            messages.extend(self.conversation_history)  # Include full conversation

            # Include the new evaluation prompt
            messages.append({"role": "user", "content": evaluation_prompt})

            if LOCAL_SETTINGS:
                response = self.llm.complete(evaluation_prompt)
                assistant_reply = dict(response)['text']
            else:
                response = self.llm.chat.completions.create(
                    model=self.llm_name,
                    messages=messages,
                )
                assistant_reply = response.choices[0].message.content

            # Save the assistant reply for further improvement attempts
            self.conversation_history.append({"role": "assistant", "content": assistant_reply})

            return assistant_reply

        # Otherwise, it's a new question
        context = self.generate_context(query)
        prompt = self.qa_prompt_tmpl_str.format(context=context, difficulty=difficulty, query=query)

        messages = [
            {"role": "system", "content": "You are a university examiner."},
            {"role": "user", "content": prompt}
        ]

        if LOCAL_SETTINGS:
            response = self.llm.complete(prompt)
            assistant_reply = dict(response)['text']
        else:
            response = self.llm.chat.completions.create(
                model=self.llm_name,
                messages=messages
            )
            assistant_reply = response.choices[0].message.content

        # Update conversation history
        self.conversation_history = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_reply}
        ]
        self.last_question = assistant_reply

        return assistant_reply

