# Collection of prompts used throughout the application

SOLUTION_RECOMMENDATION_PROMPT = """
Issue:
{issue_title}

Description:
{issue_description}

Task:
Analyze the given issue and description to generate a fully detailed solution recommendation. Your response **must be exhaustive and self-sufficient**, ensuring the user does not need to ask further questions. Provide precise technical guidance and a **fully working, production-ready code implementation**. 

**Response Format:**
1. **Understanding the Issue**: Provide an in-depth technical breakdown of the problem based on the title and description.
2. **Possible Causes**: Enumerate **all possible technical reasons** for this issue, including edge cases.
3. **Solution Recommendations**: Provide a **step-by-step, well-structured** resolution, explaining each step clearly.
4. **Code Implementation**:
    - **Provide a complete, working example** that directly addresses the issue.
    - The code must follow **industry best practices, be optimized, and production-ready**.
    - Include **necessary imports, dependencies, and setup steps**.
    - If applicable, include **test cases or validation methods**.
5. **Alternative Solutions**: If multiple approaches exist, briefly compare them and explain when each should be used.
6. **Best Practices & Optimization**: Suggest any **improvements, optimizations, or design patterns** that could enhance the implementation.

**Important Notes:**
- Do NOT provide vague or generic answers. Responses must be **highly detailed, direct, and immediately applicable**.
- Ensure **code snippets are self-contained, properly formatted, and easy to integrate**.
- If the issue requires debugging, provide **exact debugging techniques and tools** to identify the root cause.
- Avoid theoretical discussions—focus on **concrete implementations**.
"""

# CHAT_WITH_CODEBASE_PROMPT = """
# Given the following context extracted from the codebase and the user's question, generate a detailed and technically accurate response.
# Focus solely on source code files (e.g., .js, .py, .java, etc.) and avoid non-code artifacts such as migration files, package-lock files, or other configuration and generated files.
# Ensure your answer is actionable, well-supported, and technically sound. Provide code references or code suggestions as much as possible.

# Context:
# {context}

# Question:
# {question}

# Your response should:
# 1. Directly address the user's question with clear technical explanations.
# 2. Reference specific parts of the code (e.g., file names, functions, line numbers) when applicable.
# 3. Explain relevant technical concepts in a way that is accessible to both experienced developers and those less familiar with the code.
# 4. Include examples or code snippets to illustrate your points when necessary.
# 5. Suggest best practices, potential improvements, or identify any pitfalls in the code.
# 6. Ask clarifying questions if any part of the query seems ambiguous or if additional context might be required.
# 7. Clearly state any assumptions made during your analysis.

# Provide a comprehensive, structured, and easy-to-follow answer that fulfills these requirements.
# """

CHAT_WITH_CODEBASE_PROMPT = """
Given the following context extracted from the codebase and the user's question, generate a detailed and technically accurate response.
Focus solely on source code files (e.g., .js, .py, .java, etc.) and avoid non-code artifacts such as migration files, package-lock files, or other configuration and generated files.
Ensure your answer is actionable, well-supported, and technically sound, while adhering to our company's coding standards and voice.

{company_standards_section}
Context:
{context}

Question:
{question}

Your response should:
1. Directly address the user's question with clear technical explanations.
2. Reference specific parts of the code (e.g., file names, functions, line numbers) when applicable.
3. Explain relevant technical concepts in a way that is accessible to both experienced developers and those less familiar with the code.
4. Include examples or code snippets to illustrate your points when necessary.
5. Adhere to the company's coding standards, style, and voice in all explanations and suggestions.
6. Suggest best practices, potential improvements, or identify any pitfalls in the code.
7. Ask clarifying questions if any part of the query seems ambiguous or if additional context might be required.
8. Clearly state any assumptions made during your analysis.

Provide a comprehensive, structured, and easy-to-follow answer that fulfills these requirements.
"""
