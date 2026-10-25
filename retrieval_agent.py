from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool  # Assuming tools.py is in the same directory


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    text: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert research agent.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm, 
    prompt=prompt,
    tools=tools

)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

go_again = True

while go_again:
    raw_response = agent_executor.invoke({"query": input("What is your research question:")})
    #print(raw_response)
    try:
        structured_response = parser.parse(raw_response.get("output"))
        print("Topic:", structured_response.topic)
        print("\n")
        print(structured_response.text)
        print("\n")
        print("In conclusion:", structured_response.summary)
        print("\n")
        print("The sources I used:", structured_response.sources)
    except Exception as e:
        print("Error parsing response:", e, "Raw response:", raw_response)

    go_again = input("Do you have another question? (y/n): ").lower()

    if go_again == 'y':
        go_again = True
    else:
        go_again = False
        print("Goodbye! Thank you for using the research agent!")