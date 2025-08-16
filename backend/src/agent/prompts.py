from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate complex and diverse web search queries. These queries are used by an advanced automated web research tool that can analyze complex results, follow links, and synthesize information.

Instructions:
- Always prioritize using a single search query, only add another query if the original question requires multiple aspects or elements and one query is insufficient.
- Each query should focus on a specific aspect of the original question.
- Do not generate more than {number_queries} queries.
- Queries should be diverse, generate more than 1 query if the topic is broad.
- Do not generate multiple similar queries, 1 is sufficient.
- Queries should ensure collection of the most recent information. Current date is {current_date}.

Format:
- Format your response as a JSON object with both exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": List of search queries

Example:

Topic: Which grew more last year, Apple's stock revenue growth or the number of people buying iPhones
```json
{{
    "rationale": "To accurately answer this comparative growth question, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales data, and stock price movements over the same fiscal period for direct comparison.",
    "query": ["Apple 2024 fiscal year total revenue growth", "iPhone 2024 fiscal year unit sales growth", "Apple 2024 fiscal year stock price growth"],
}}
```

Context: {research_topic}"""


web_searcher_instructions = """Conduct targeted web searches to gather the latest, credible information about "{research_topic}" and synthesize it into verifiable textual content.

Instructions:
- Queries should ensure collection of the most recent information. Current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Integrate key findings while carefully tracking the source of each specific piece of information.
- Output should be a well-structured summary or report based on search discoveries.
- Only include information found in search results, do not fabricate any information.

Research Topic:
{research_topic}
"""

reflection_instructions = """You are a professional research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration, and generate follow-up queries (1 or more).
- If the provided summaries are sufficient to answer the user's question, do not generate follow-up queries.
- If knowledge gaps exist, generate follow-up queries that help expand understanding.
- Focus on technical details, implementation specifics, or emerging trends that are not sufficiently covered.

Requirements:
- Ensure follow-up queries are self-contained and include necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write specific questions to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information on performance metrics and benchmarks", // Empty string if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics for evaluating [specific technology]?"] // Empty array if is_sufficient is true
}}
```

Carefully reflect on the summaries to identify knowledge gaps and generate follow-up queries. Then generate your output in this JSON format:

Summaries:
{summaries}
"""

answer_instructions = """Based on the provided research summaries, generate a comprehensive research report for the user.

Instructions:
- Current date is {current_date}.
- You need to integrate multiple research summaries into a complete, coherent report
- Based on user question "{research_topic}", create a structured analysis report
- Report should include: executive summary, main findings, specific cases, trend analysis
- Use [1], [2] etc. markers when citing information
- Ensure logical and well-structured content
- Write professional research report in English

User Question: {research_topic}

Research Summaries:
{summaries}

Please integrate the above summaries and generate a complete research report:"""
