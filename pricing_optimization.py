import yaml
import logging
from typing import Sequence, List
from langchain_groq import ChatGroq
from langchain.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain.schema import AgentAction
from langgraph.graph import END, MessageGraph
import backoff

from constants import GROQ_API_KEY, INSTRUCTIONS, REFLECTION_SYSTEM_MESSAGE
from utils import extract_python_code, extract_final_output

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def create_llm(api_choice='groq'):
    if api_choice == 'groq':
        return ChatGroq(temperature=config['temperature'],
                        model=config['models']['agent'],
                        api_key=GROQ_API_KEY)
    elif api_choice == 'ollama':
        return Ollama(model=config['models']['ollama'])
    else:
        raise ValueError(f"Invalid API choice: {api_choice}")

tools = [PythonREPLTool()]

base_prompt = PromptTemplate.from_template("""
{instructions}

TOOLS:
------
You have access to the following tools:
{tools}

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!
messages: {messages}
{agent_scratchpad}
""")

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", REFLECTION_SYSTEM_MESSAGE),
    MessagesPlaceholder(variable_name="messages"),
])

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generation_node(state: Sequence[BaseMessage]):
    try:
        logger.info("Generating pricing recommendation")
        result_generate = generate.invoke({"messages": state})
        logger.debug(f"Generation result: {result_generate}")

        python_code = ""
        final_output = ""
        for step in result_generate['intermediate_steps']:
            if isinstance(step[0], AgentAction) and step[0].tool == "Python_REPL":
                python_code = step[0].tool_input
                final_output = step[1]

        if not python_code and not final_output:
            final_output = result_generate['output']

        logger.info(f"Generated Python code: {python_code}")
        logger.info(f"Generated pricing recommendation: {final_output}")

        combined_output = f"Python code:\n```python\n{python_code}\n```\nPricing recommendation: {final_output}"
        return AIMessage(content=combined_output)
    except Exception as e:
        logger.error(f"Error in generation node: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    try:
        logger.info("Reflecting on pricing recommendation")
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [messages[0]] + [cls_map[msg.type](content=msg.content) for msg in messages[1:] if
                                      isinstance(msg, BaseMessage)]

        if not translated:
            logger.warning("No valid messages to reflect on")
            return [HumanMessage(content="No valid pricing recommendation to reflect on.")]

        res = reflect.invoke({"messages": translated})

        if isinstance(res, list):
            reflection_content = ' '.join([str(item) for item in res])
        elif isinstance(res, str):
            reflection_content = res
        elif hasattr(res, 'content'):
            reflection_content = res.content
        else:
            reflection_content = str(res)

        logger.info(f"Reflection output: {reflection_content}")
        return [HumanMessage(content=reflection_content)]
    except Exception as e:
        logger.error(f"Error in reflection node: {str(e)}")
        raise

def should_continue(state):
    if not state:
        return "end"
    last_message = state[-1]
    if isinstance(last_message, BaseMessage) and "optimal" in last_message.content.lower() and "strategy" in last_message.content.lower():
        return "end"
    if len(state) > config['max_iterations']:
        logger.warning("Maximum iterations reached, ending execution.")
        return "end"
    return "generate"

def run_pricing_optimization(market_data, api_choice='groq'):
    try:
        llm = create_llm(api_choice)
    except Exception as e:
        logger.error(f"Failed to initialize LLM with {api_choice} API: {str(e)}")
        return None, None, None

    prompt_agent = base_prompt.partial(instructions=INSTRUCTIONS)
    agent = create_react_agent(llm, tools, prompt_agent)
    global generate, reflect
    generate = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True,
                             handle_parsing_errors=True)
    reflect = reflection_prompt | llm

    builder = MessageGraph()
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.set_entry_point("generate")
    builder.add_conditional_edges("reflect", should_continue, {"generate": "generate", "end": END})
    builder.add_edge("generate", "reflect")
    graph = builder.compile()

    request = HumanMessage(f"Generate a pricing recommendation based on the following market data: {market_data}")

    try:
        results = []
        for event in graph.stream([request]):
            if 'generate' in event:
                results.append(event['generate'])
            if 'reflect' in event:
                results.append(event['reflect'][0] if isinstance(event['reflect'], list) else event['reflect'])
            if event.get('__end__'):
                break

        if results:
            final_output = next((r.content for r in reversed(results) if isinstance(r, AIMessage)),
                                "No output generated.")
            python_code = extract_python_code(final_output)
            pricing_recommendation = extract_final_output(final_output)
            return results, python_code, pricing_recommendation
        else:
            return None, None, None

    except Exception as e:
        logger.error(f"Error during pricing optimization: {str(e)}")
        return None, None, None