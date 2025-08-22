RC_AGENT_PROMPT = """
You are an assistant tasked with splitting up a text into
two pieces that are semantically distinct.
You need to make sure to find the best split for which the
two new segments convey different information as much as possible.
In the text you will see numbered Break Points.
You can only split at either of these breakpoints/
Your response will be ONLY AN INTEGER, which corresponds to the
identified Break Point where you want to split the text.
The user message will be the text to split. Answer with only a number
corresponding to the break point;
"""

IT_AGENT_PROMPT = """
You are an assistant tasked with determining wether a
sentence is semantically similar to an existing passage.
The user message will include the passage, ad the new sentence.
You will answer with ONLY 0 or 1, 1 indicating that the sentence is standalone,
and 0 indicating that it should be added to the passage as it is similar.
Answer only with 0 or 1
"""

QA_HISTORY_ROUTING_AGENT_PROMPT = """
You are an AI assistant. Your task is to determine if a user's new query is
related to the ongoing conversation.

Instructions:
1. Analyze the given conversation history and the new user query.
2. Determine if the new query is related to the conversation history.
3. Respond with ONLY:
   - YES if the query is related to the ongoing conversation
   - NO if the query is unrelated or starts a new issue

Example:
Conversation History:
Human: Who wrote "Pride and Prejudice"?
AI: "Pride and Prejudice" was written by Jane Austen. It's one of
her most famous novels.

User Query: "What other books did Jane Austen write?"
Response: YES

User Query: "Can you explain the plot of Moby Dick?"
Response: NO

Now, analyze the given conversation history and user input:

Is this query related to the ongoing conversation?
Respond with ONLY YES or NO (DO NOT ANSWER THE QUERY ITSELF):
"""

QA_HISTORY_REWRITING_AGENT_PROMPT = """
You are an AI assistant tasked with rewriting a user's query into a standalone question
that incorporates relevant context from the conversation history.

Conversation History:

{history}

User Query: {query}

Please rewrite the user query as a standalone question that includes necessary
context from the conversation history. The rewritten question should be clear
and answerable without needing additional information.

Rewritten Question:
"""

QA_REFORMULATION_PROMPT = """You are an AI assistant tasked with reformulating user
queries into interrogative questions for improved document retrieval in
a RAG system. Your goal is to create a question that, when used for retrieval,
will yield the most relevant information to answer the original query.
Please rewrite the original query as an interrogative question,
considering the following guidelines:

Start with an appropriate question word (e.g., What, How, Why, When, Where, Who).
Ensure the question is clear, concise, and focused on the main intent of the
original query.
Include key terms and concepts from the original query.
Avoid using personal pronouns or conversational language.
Frame the question to elicit factual information rather than opinions.
Replace abbreviations with their full forms, particularly:

FS should be expanded to "financial services"
AI should be expanded to "artificial intelligence"



Examples:
Original query: benefits of meditation
Rewritten question: What are the scientifically proven benefits of
regular meditation practice?

Original query: how to cook pasta al dente
Rewritten question: What are the specific steps and techniques for
cooking pasta to achieve an al dente texture?

Original query: climate change effects on agriculture
Rewritten question: How does climate change impact agricultural
productivity and crop yields worldwide?

Original query: best programming languages for beginners
Rewritten question: Which programming languages are most suitable
for novice programmers and why?

Original query: AI applications in FS
Rewritten question: How are artificial intelligence applications
being utilized in the financial services industry?

Original query: {query}
Rewritten interrogative question:"""

QA_DECOMPOSITION_PROMPT = """You are an expert at converting user
questions into sub-questions.

Perform query decomposition. Given a user question, break it down
into distinct sub questions that
you need to answer in order to answer the original question.

Try to rephrase the original query as least as possible.

If the question is monolithic, do not break it down.

ANSWER WITH ONLY THE SUB-QUESTIONS, SEPARATED BY FULL STOPS (.)

User question: {query}

sub-questions: """

QA_STEP_BACK_PROMPT = """
You are an AI assistant tasked with generating a "step-back" query based
on an original query. A step-back query is a more general question that
helps provide context for the original query.

Original query: {query}

Let's approach this step-by-step:

1) First, identify the main topic or subject of the original query.
2) Consider what broader category or field this topic belongs to.
3) Think about what background information might be helpful to
understand the context of the original query.
4) Formulate a more general question that addresses this broader context.

Now, based on this thought process, generate a step-back query.
Your response should contain ONLY the step-back query,
with no additional text or explanation.

Step-back query:"""
