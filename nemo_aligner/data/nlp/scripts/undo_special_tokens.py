# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to remove special tokens from dpo datasets 
and convert them into list of messages format"""

import json
import re
import sys

input_jsonl = sys.argv[1]
output_jsonl = input_jsonl.replace(".jsonl", ".no_special_toks.jsonl")


def format_conversation(input_string):
    # Define roles and patterns
    role_patterns = {"<extra_id_0>System": "system", "<extra_id_1>User": "user", "<extra_id_1>Assistant": "assistant"}

    # Initialize an empty output list
    conversation = []

    # Use regex to find each segment's role and content
    segments = re.findall(r"(<extra_id_[0-1]>[^\n]+)\n(.*?)((?=<extra_id_)|$)", input_string, re.DOTALL)

    for segment in segments:
        role_tag, content, _ = segment
        role = role_patterns.get(role_tag.strip(), "unknown")
        conversation.append({"role": role, "content": content.strip()})

    empty_asst = conversation.pop()

    return conversation, empty_asst


with open(input_jsonl, "r") as f, open(output_jsonl, "w") as w:
    for line in f:
        j = json.loads(line)
        prompt = j["prompt"]
        undo_spl_prompt, empty_assistant = format_conversation(prompt)
        chosen, rejected = j["chosen_response"], j["rejected_response"]
        chosen = chosen.split("\n<extra_id_1>")[0]
        rejected = rejected.split("\n<extra_id_1>")[0]
        chosen_message = {"role": empty_assistant["role"], "content": chosen}
        rejected_message = {"role": empty_assistant["role"], "content": rejected}
        j_out = {
            "prompt": undo_spl_prompt,
            "chosen_response": chosen_message,
            "rejected_response": rejected_message,
            "chosen_reward": j["chosen_reward"],
            "rejected_reward": j["rejected_reward"],
        }
        w.write(json.dumps(j_out) + "\n")

old_format_example = """<extra_id_0>System\n\n<extra_id_1>User\nDoes GPT-4 use RNN or Transformer models, and which one is better for this type of project?\n<extra_id_1>Assistant\nGPT-4 uses a transformer architecture, not a recurrent neural network. Both models are commonly used for natural language processing tasks, and both have advantages and disadvantages, so it depends on the specific application of the model.\n<extra_id_1>User\nCould you explain in detail both the advantages and disadvantages from different perspectives?\n<extra_id_1>Assistant\nYes, here are a few different perspectives on the pros and cons of RNN and transformer models:\n \n\n The RNN model has better sequential memory and can better model sequences of data, whereas the transformer model has better parallel computation and can better model large input datasets. This makes RNNs better for applications like speech recognition and machine translation, where the input is a sequence of sounds or words, and transformers better for applications like natural language processing and text classification, where the input is a large collection of documents or social media posts.\n \n\n RNNs are more straightforward and easier to understand and debug, whereas transformers are more flexible and can be used for more diverse applications. This makes RNNs better for teaching and research, and transformers better for actual applications in industry and the real world.\n \n\n RNNs are more efficient and faster to run, whereas transformers are more accurate and produce better results. This makes RNNs better for fast prototyping and testing, and transformers better for actual deployment in production applications.\n<extra_id_1>User\ncould you add more in a table\n<extra_id_1>Assistant\nHere are some more pros and cons of RNN and transformer models:\n \n\n Pros of RNNs:\n \n\n -  Better sequential memory and handling of sequences\n -  Simpler and easier to understand\n -  Faster to run\n -  Good for teaching and research\n \n\n Cons of RNNs:\n \n\n -  Limited parallelism\n -  Poor handling of large input datasets\n -  Not as accurate as transformers\n -  Not as flexible as transformers\n \n\n Pros of transformers:\n \n\n -  Better parallelism and handling of large input datasets\n -  More flexible and versatile\n -  Higher accuracy\n -  Good for actual applications in industry and the real world\n \n\n Cons of transformers:\n \n\n -  More complex and harder to understand and debug\n -  Slower to run\n -  Not as good for teaching and research\n<extra_id_1>User\nFor a chat-based music recommendation platform, which model is recommended? Please mention any model that works well, including those mentioned above or any others.\n<extra_id_1>Assistant\nFor a chat-based music recommendation platform, I would recommend using a transformer architecture, such as the GPT-4 model. Transformers are better for handling large input datasets, and they are more flexible and versatile than RNNs. They are also more accurate, so they are better for actually recommending music to users in a real-world application. RNNs are better for teaching and research, as they are simpler and easier to understand and debug.\n<extra_id_1>User\ncould you add more?\n<extra_id_1>Assistant\n"""

print("****************************************")
print("old format prompt")
print("****************************************")
print(old_format_example)
print("****************************************")
print("new format prompt")
print("****************************************")
new_format_example, _ = format_conversation(old_format_example)
print(json.dumps(new_format_example, indent=2))
