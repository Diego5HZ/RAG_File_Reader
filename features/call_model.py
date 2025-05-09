# LLM calls and system prompt

import ollama

system_prompt = """
You are an AI research assistant specializing in scientific and technical analysis. 
Your task is to generate well-structured, methodical, and precise responses based solely on the given context. 
Only use information present in the context—do not make assumptions or use external knowledge.

Respond in a clear, scientific style using:
1. Paragraphs or bullet points for structure.
2. Proper grammar and concise explanations.
3. Headings or subheadings if needed.
"""

# system_prompt = """
# You are an AI research assistant specializing in scientific and technical analysis. 
# Your task is to generate well-structured, methodical, and precise responses based solely on the given context. 
# Approach the question using analytical reasoning, evidence from the context, and principles of scientific communication.

# When answering:
# 1. Identify and extract key data points from the context.
# 2. Apply logical reasoning to connect relevant findings to the research question.
# 3. Structure your response clearly, with sections where applicable.
# 4. Only use information present in the context—do not make assumptions or use external knowledge.
# 5. Present your findings as if preparing a report for a scientific audience.

# Format your response as follows:
# 1. Use clear, concise language.
# 2. Organize your answer into paragraphs for readability.
# 3. Use bullet points or numbered lists where appropriate to break down complex information.
# 4. If relevant, include any headings or subheadings to structure your response.
# 5. Ensure proper grammar, punctuation, and spelling throughout your answer.

# Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
# """

def call_llm(context: str, prompt: str, max_context_chars: int = 12000):
    """
    Calls Ollama LLM with formatted context and prompt.

    Args:
        context (str): Retrieved document content.
        prompt (str): User question.
        max_context_chars (int): Max length of context passed to LLM (default: 12,000 chars).

    Yields:
        str: Streamed response chunks from the LLM.
    """
    try:
        context = context.strip()
        prompt = prompt.strip()

        if len(context) > max_context_chars:
            context = context[:max_context_chars]
            context += "\n\n[Context truncated due to length.]"

        full_user_prompt = f"""Answer the following question using ONLY the provided context.

Context:
\"\"\"
{context}
\"\"\"

Question: {prompt}
"""

        response = ollama.chat(
            model="mistral:latest",
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_prompt},
            ],
        )

        for chunk in response:
            if chunk.get("done") is False:
                yield chunk["message"]["content"]
            else:
                break

    except Exception as e:
        yield f"❌ LLM error: {str(e)}"