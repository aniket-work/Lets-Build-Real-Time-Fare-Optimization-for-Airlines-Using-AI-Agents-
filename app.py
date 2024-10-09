import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain.schema import AgentAction
from typing import List, Sequence
from langgraph.graph import END, MessageGraph
import re
import yaml
import logging
import random
import backoff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and configuration
load_dotenv()
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# Initialize LLM with retry mechanism
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def create_llm(api_choice='groq'):
    if api_choice == 'groq':
        api_key = os.getenv('GROQ_API_KEY')
        return ChatGroq(temperature=config['temperature'],
                        model=config['models']['agent'],
                        api_key=api_key)
    elif api_choice == 'ollama':
        return Ollama(model=config['models']['ollama'])
    else:
        raise ValueError(f"Invalid API choice: {api_choice}")


# Set up tools
tools = [PythonREPLTool()]

# Define prompt templates
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

instructions = """You are an AI agent designed to generate and optimize pricing strategies for an e-commerce platform or airline.
Your task is to analyze market data, competitor pricing, and demand forecasts to generate pricing recommendations.
Then, reflect on these recommendations considering factors like profit margins, customer retention, and long-term market positioning.
Use Python to perform calculations and data analysis. If you need to generate random data for simulation, you can do so.
Iterate on the pricing strategy until you find an optimal balance.
"""

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in pricing strategy and market analysis. "
               "Evaluate the provided pricing recommendation and consider its impact on profit margins, customer retention, and long-term market positioning. "
               "If the strategy needs improvement, explain why and suggest areas for refinement. "
               "If the strategy seems optimal, confirm its suitability."),
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
            # If no Python code was executed, use the final answer as the output
            final_output = result_generate['output']

        logger.info(f"Generated Python code: {python_code}")
        logger.info(f"Generated pricing recommendation: {final_output}")

        # Combine the Python code and the final output into a single message
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

        # Ensure we have messages to reflect on
        if not translated:
            logger.warning("No valid messages to reflect on")
            return [HumanMessage(content="No valid pricing recommendation to reflect on.")]

        res = reflect.invoke({"messages": translated})

        # Check if res is a list, string, or has a content attribute
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
    if isinstance(last_message,
                  BaseMessage) and "optimal" in last_message.content.lower() and "strategy" in last_message.content.lower():
        return "end"
    if len(state) > 10:
        logger.warning("Maximum iterations reached, ending execution.")
        return "end"
    return "generate"


def extract_python_code(text):
    pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else "No Python code found."


def extract_final_output(text):
    pattern = re.compile(r'Pricing recommendation: (.*)')
    match = pattern.search(text)
    return match.group(1).strip() if match else "No pricing recommendation found."


def main(api_choice='groq'):
    try:
        llm = create_llm(api_choice)
    except Exception as e:
        logger.error(f"Failed to initialize LLM with {api_choice} API: {str(e)}")
        return

    prompt_agent = base_prompt.partial(instructions=instructions)
    agent = create_react_agent(llm, tools, prompt_agent)
    global generate, reflect
    generate = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True,
                             handle_parsing_errors=True)
    reflect = reflection_prompt | llm

    # Build the graph
    builder = MessageGraph()
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.set_entry_point("generate")
    builder.add_conditional_edges("reflect", should_continue, {"generate": "generate", "end": END})
    builder.add_edge("generate", "reflect")
    graph = builder.compile()

    # Simulated market data
    market_data = {
        "average_price": 100,
        "competitor_prices": [95, 98, 102, 105],
        "demand_forecast": 1000,
        "current_inventory": 1200,
    }

    request = HumanMessage(f"Generate a pricing recommendation based on the following market data: {market_data}")

    try:
        results = []
        for event in graph.stream([request]):
            print('.', end='', flush=True)
            logger.debug(f"Event: {event}")
            if 'generate' in event:
                results.append(event['generate'])
            if 'reflect' in event:
                results.append(event['reflect'][0] if isinstance(event['reflect'], list) else event['reflect'])
            if event.get('__end__'):
                break
            time.sleep(1)  # Add a delay between requests

        print()  # New line after progress dots

        if results:
            for i, result in enumerate(results):
                logger.info(f"Step {i + 1}: {result.content}")

            final_output = next((r.content for r in reversed(results) if isinstance(r, AIMessage)),
                                "No output generated.")

            python_code = extract_python_code(final_output)
            pricing_recommendation = extract_final_output(final_output)

            print("Final Python Code:")
            print(python_code if python_code != "No Python code found." else "No Python code was generated.")
            print("\nFinal Pricing Recommendation:")
            print(
                pricing_recommendation if pricing_recommendation != "No pricing recommendation found." else final_output)

            logger.info(f"Final output: Python Code: {python_code}, Pricing Recommendation: {pricing_recommendation}")
        else:
            print("The pricing optimization process did not complete as expected.")
            logger.warning("Graph execution completed without results.")

    except Exception as e:
        logger.error(f"Error during pricing optimization: {str(e)}")
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import sys

    api_choice = sys.argv[1] if len(sys.argv) > 1 else 'groq'
    main(api_choice)