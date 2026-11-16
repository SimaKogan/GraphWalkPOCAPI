import numpy as np
from google.genai import types
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from Core.Agents.base_agent import BaseAgent
from Core.Utils.llm_utils import LLMOptions, load_prompt_config, format_decimals_in_string


class PromptBasedAgent(BaseAgent):
    def __init__(self, name: str, prompt_type: str, llm_options: LLMOptions | None = None):
        super().__init__(name=name, llm_options=llm_options)
        self.prompt_type = prompt_type
        self.prompt_config = load_prompt_config(prompt_type)
        self.system_instruction = self.prompt_config['instructions']

    def prompt_based_generate(self, conv_history=None, format_decimals=4, **params) -> str:
        """
        Generate a response using the configured prompt type

        :param conv_history: Conversation history to pass to the LLM
        :param format_decimals: Number of decimals to format the response
        :param params: Parameters to substitute into the prompt template
                    (keys should match those in prompt_params from the config)
        :return: Generated response
        """
        # Validate required params
        missing_params = set(self.prompt_config['prompt_params']) - set(params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Format the prompt
        try:
            formatted_prompt = self.prompt_config['prompt'].format(**params)
        except KeyError as e:
            raise ValueError(f"Invalid parameter in prompt template: {e}")

        if format_decimals is not None:
            formatted_prompt = format_decimals_in_string(formatted_prompt)

        contents = []
        # Add conversation history if provided
        if conv_history:
            for msg in conv_history:
                contents.append(types.Content(role=msg["role"], parts=[types.Part.from_text(text=str(msg["text"]))]))

        # Add the new user prompt
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=formatted_prompt)]))
        return self.generate(contents,
                             system_instruction=self.prompt_config['instructions'])

    def _temps_options(self, max_retries=5, start=0.1, end=0.6):
        return np.linspace(start, end, max_retries).tolist()

    def _models_options(self):
        return ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.5-pro"]

    def get_response_and_parse(self, output_parser, max_retries=5, conv_history=None, format_decimals=None, **kwargs):
        """
        Generate a response and parse it using the provided output parser.
        Max retries are used to handle errors during generation and parsing.
        Max retries is the number of attempts with different temperatures per each model.
        Iterate through the temperature options and then models.

        :param conv_history: Conversation history to pass to the LLM
        :param output_parser: The output parser to use
        :param max_retries: The maximum number of retries per model - number of temperature options
        :param kwargs: Additional keyword arguments
        :return: The parsed response
        """
        response_dict = None
        parsed_correctly = False
        retries = 0
        # Temperature options
        original_temp = self.llm_options.temp
        temps = self._temps_options(start=original_temp, max_retries=max_retries)

        # Model options
        original_model = self.llm_options.model_name
        models = self._models_options()
        if original_model in models:
            models.remove(original_model)

        while not parsed_correctly and retries < max_retries:
            try:
                self.llm_options.temp = temps[retries]
                response_prompt = self.prompt_based_generate(conv_history=conv_history, format_decimals=format_decimals,
                                                             **kwargs)
                response_dict = output_parser.parse(response_prompt)
                parsed_correctly = True
            except Exception as e:  # fail-safe for parsing errors
                retries += 1
                print(e)
                if retries == max_retries:
                    if len(models) > 0:
                        self.llm_options.model_name = models.pop()
                        retries = 0
                        self.llm_options.temp = temps[retries]
        # Reset to original values
        self.llm_options.temp = original_temp
        self.llm_options.model_name = original_model
        if not parsed_correctly:
            raise ValueError("Failed to generate or parse the response")
        return response_dict

    def get_output_parser(self):
        output_schema = self.prompt_config['output_schema']
        response_schemas = []
        for schema in output_schema:
            response_schemas.append(ResponseSchema(
                name=schema['name'],
                description=(schema['description']),
                type=schema['type']
            ))
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        return format_instructions, output_parser

    @staticmethod
    def obj_to_json(obj):
        import json
        # Use single quotes within the tuple to avoid f-string parsing edge-cases.
        return f"```json\n{json.dumps(obj, ensure_ascii=False, separators=(',', ':'), indent=2)}\n```"
