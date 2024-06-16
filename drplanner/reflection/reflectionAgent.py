import os
import textwrap
import time
from drplanner.utils.config import DrPlannerConfiguration
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage



class ReflectionAgent:
    def __init__(
        self, temperature: float = 0.0, verbose: bool = False
    ) -> None: 
        #todo:change the instantiate according to config
        self.config = DrPlannerConfiguration()
        self.llm = ChatOpenAI(
            openai_api_key=self.config.openai_api_key,
            temperature=temperature,
            model_name=self.config.gpt_version,
            max_tokens=3000,
            request_timeout=60,
        )


    def reflection(self, human_message: str, llm_response: str):
        delimiter = "####"
        system_message = textwrap.dedent(f"""\
        You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        You will be given a detailed description of the motion planner and heuristic function of current solution. 

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output identify diagnoses and recommend prescriptions for the motion planner> 

        Make sure to include {delimiter} to separate every step.
        """)
        human_message = textwrap.dedent(f"""\
            ``` Human Message ```
            {human_message}
            ``` ChatGPT Response ```
            {llm_response}

            Now, you know this prescription ChatGPT output cause too much costs or do not work after taking this prescription, which means there are some mistake in ChatGPT resoning and cause the wrong action.    
            Please carefully check every reasoning in ChatGPT response and find out the mistake in the reasoning process of ChatGPT, and also output your corrected version of ChatGPT response.
            Your answer should use the following format:
            {delimiter} Analysis of the mistake:
            <Your analysis of the mistake in ChatGPT reasoning process>
            {delimiter} What should ChatGPT do to avoid such errors in the future:
            <Your answer>
            {delimiter} Corrected version of ChatGPT response:
            <Your corrected version of ChatGPT response>
        """)

        print("Self-reflection is running, may take time...")
        start_time = time.time()
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]
        response = self.llm(messages)
        target_phrase = f"{delimiter} What should ChatGPT do to avoid such errors in the future:"
        substring = response.content[response.content.find(
            target_phrase)+len(target_phrase):].strip()
        corrected_memory = f"{delimiter} I have made a misake before and below is my self-reflection:\n{substring}"
        print("Reflection done. Time taken: {:.2f}s".format(
            time.time() - start_time))
        print("corrected_memory:", corrected_memory)

        return corrected_memory
