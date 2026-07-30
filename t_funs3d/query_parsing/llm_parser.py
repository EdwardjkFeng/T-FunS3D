import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig

import json
import time

class LLMParser:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-4B",
                 torch_dtype: str = "auto",
                 device_map: str = "auto",
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only = False,
        )
        

        self.system_prompt1 = """You are an AI system that generates JSON instructions for a robot to identify physical objects and their spatial relations based on a natural language command.  
        - Use only **concrete physical objects** (furniture, appliances, manipulable items).  
        - Do NOT use rooms (e.g., kitchen, bedroom) as objects or in relations.  
        - Retain descriptive adjectives of objects (e.g., "blue chair", "wooden desk").  
        - The only allowed spatial relations are: [in, on, under, around, next to, near, over, behind, in_front_of, to_the_left, to_the_right].  
        - Relations must be **between visible or manipulable objects only**.  
        - Output must be **valid JSON only** (no text before or after).  
        - JSON rules: double quotes for keys/strings, lowercase `true`/`false`, `null` if no value, lists as JSON arrays.  
        """
        
        self.system_prompt2 = """You are an AI system that generates JSON instructions for a robot to decompose a task into actions on physical objects.  
        - Retain adjectives describing object properties (e.g., "red button", "metal handle", "right socket“， ”left door").  
        - The only allowed robot actions are: [rotate, key_press, tip_push, hook_pull, pinch_pull, hook_turn, foot_push, plug_in, unplug].  
        - The acted_on_object must be the **lowest-level physical part** that affords the action (e.g., handle, knob, button, latch). 
        - Do not stop at container objects like "drawer" or "door" if they normally require a part (e.g., handle) to interact with. 
        - The acted_on_object_hierarchy must always begin with the top-level object followed by its parts (e.g., ["closet", "closet door", "handle"]).
        - Object hierarchies must describe **part-of relations only** (e.g., [cabinet, handle]). 
        - Do NOT include objects that are merely adjacent or spatially related.  
        - Output must be **valid JSON only**.  
        - JSON rules: double quotes for keys/strings, lowercase `true`/`false`, `null` if no value, lists as JSON arrays.  
        """
        
        self.statement1 = """To {query}, what do I know?  
        Respond with valid JSON only in the following format:
        
        {{
          "prompt": "{query}",
          "space": "room or space where the object is located, if explicitly mentioned; otherwise null",
          "spatial_relation": [
            "Chain of spatial relations ONLY between physical objects (e.g., 'cabinet is under the TV').",
            "If no relations exist, return an empty list.",
            "If the natural language does not explicitly specify a relation, do not infer one. Use near only if adjacency is clearly implied.",
            "It is better to output fewer or no relations than to invent uncertain or incorrect ones."
          ],
          "referent_object_hierarchy": [
            "List of physical objects used to locate the acted-on part (excluding rooms).",
            "Follow the order based on 'spatial_relation'."
          ]
        }}
        """
        
        self.statement2 = """I know {spatial_relations}. How do I {query} step by step?  
        Respond with valid JSON only in the following format:
        
        {{
          "prompt": "{query}",
          "task_solving_sequence": [
            "Step-by-step subtasks as natural language strings."
          ],
          "acted_on_object": "the lowest level part to act on (e.g., handle, button, knob) with the allowed robot actions.",
          "acted_on_object_hierarchy": [
            "A list from the top-level object to its 'acted_on_object'.",
            "Always include the main object (e.g., closet, cabinet, desk)."
            "Do NOT include referent objects",
            "Then add the part (e.g., left closet door), and finally the actionable sub-part (e.g., handle)."
          ]
        }}
        """


    def parse_query(self, query: str, max_new_tokens: int = 32768, show_thinking: bool = False) -> str:

        messages = [
            {"role": "system", "content": self.system_prompt1},
            {"role": "user", "content": self.statement1.format(query=query.lower())}
        ]
    
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True, # Enable thinking for extracting relations
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        start = time.time()
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        stop = time.time()
        print(f"System respond in {stop - start:.2f}s.")
        
        response = self.show_results(output_ids)
        self.content = json.loads(response)
        self.content.pop("prompt")
        
    
        messages.append({"role": "system", "content": self.system_prompt2})
        messages.append({"role": "user", "content": self.statement2.format(spatial_relations=self.content, query=query.lower())})
    
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        start = time.time()
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        stop = time.time()
        print(f"System respond in {stop - start:.2f}s.")
    
        self.content.update(json.loads(self.show_results(output_ids)))
        print(f"[INFO] Completed query parsing.")

        # #convert the string to a dictionary
        # try:
        #     self.content = eval(self.content)
        # except SyntaxError as e:
        #     print(f"[ERROR] Failed to parse content: {self.content}")
        #     raise e
        
        # print(f"[INFO] Completed query parsing. \ncontent: {self.content}")

        return self.content

    def show_results(self, output_ids, show_thinking=False):
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
            
        if show_thinking:
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            print("thinking content:", thinking_content)
            
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print("response:", content)
        return content
    
    def get_contextual_object(self) -> str:
        """
        Get the contextual object from the query using the LLM.
        """
        contextual_object = self.content["acted_on_object_hierarchy"][0]
        return contextual_object.lower()

    def get_functional_element(self) -> str:
        """
        Get the functional element from the query using the LLM.
        """
        functional_obj = self.content["acted_on_object"]
        return functional_obj
    
    def get_referent_object_hierarchy(self) -> list:
        """
        Get the referent object hierarchy from the query using the LLM.
        """
        referent_object_hierarchy = self.content["referent_object_hierarchy"]
        return referent_object_hierarchy
    
    def get_spatial_relation(self) -> list:
        """
        Get the spatial relation from the query using the LLM.
        """
        spatial_relation = self.content["spatial_relation"]
        return spatial_relation
    
    def get_task_solving_sequence(self) -> list:
        """
        Get the task solving sequence from the query using the LLM.
        """
        task_solving_sequence = self.content["task_solving_sequence"]
        return task_solving_sequence
    
    def get_spaece(self) -> str:
        """
        Get the space from the query using the LLM.
        """
        space = self.content["space"]
        return space
    