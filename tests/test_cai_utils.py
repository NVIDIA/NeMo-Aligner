# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import sys

sys.path.append(os.path.abspath("../examples/nlp/cai"))

from cai_utils import ChatTemplateHelper, PromptTemplate, UserAssistantPromptTemplate


def test_extra_id_format_single_user_prompt():
    extra_id_prompt_template = PromptTemplate(
        dict(
            System="<extra_id_0>System\n{MESSAGE}\n",
            User="<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n",
            Assistant="{MESSAGE}\n",
        ),
        eos_token="<extra_id_1>",
        response_extract_pattern="<extra_id_1>Assistant\n",
    )
    m11 = extra_id_prompt_template.format_messages(
        [{"role": "System", "content": ""}, {"role": "User", "content": "Calculate the sum of 2 and 3."}]
    )
    m11_expected = """<extra_id_0>System

<extra_id_1>User
Calculate the sum of 2 and 3.
<extra_id_1>Assistant
"""
    assert m11 == m11_expected, "extra_id formatting failed"


def test_extra_id_format_single_message_chat():
    extra_id_prompt_template = PromptTemplate(
        dict(
            System="<extra_id_0>System\n{MESSAGE}\n",
            User="<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n",
            Assistant="{MESSAGE}\n",
        ),
        eos_token="<extra_id_1>",
        response_extract_pattern="<extra_id_1>Assistant\n",
    )
    m12 = extra_id_prompt_template.format_messages(
        [
            {"role": "System", "content": ""},
            {"role": "User", "content": "Calculate the sum of 2 and 3."},
            {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
        ]
    )
    m12_expected = """<extra_id_0>System

<extra_id_1>User
Calculate the sum of 2 and 3.
<extra_id_1>Assistant
The sum of 2 and 3 is 5.
"""
    assert m12 == m12_expected, "extra_id formatting single message chat failed"

    extract_m12 = extra_id_prompt_template.extract_response(m12)
    extract_m12_expected = "The sum of 2 and 3 is 5."
    assert extract_m12 == extract_m12_expected, "extra_id formatting single message chat extract failed"


def test_extra_id_format_multi_message_chat():
    extra_id_prompt_template = PromptTemplate(
        dict(
            System="<extra_id_0>System\n{MESSAGE}\n",
            User="<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n",
            Assistant="{MESSAGE}\n",
        ),
        eos_token="<extra_id_1>",
        response_extract_pattern="<extra_id_1>Assistant\n",
    )

    m13 = extra_id_prompt_template.format_messages(
        [
            {"role": "System", "content": ""},
            {"role": "User", "content": "Calculate the sum of 2 and 3."},
            {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
            {"role": "User", "content": "Thank you! Could you also calculate the sum of 5 and 7?"},
        ]
    )
    m13_expected = """<extra_id_0>System

<extra_id_1>User
Calculate the sum of 2 and 3.
<extra_id_1>Assistant
The sum of 2 and 3 is 5.
<extra_id_1>User
Thank you! Could you also calculate the sum of 5 and 7?
<extra_id_1>Assistant
"""
    assert m13 == m13_expected, "extra_id formatting multi message chat failed"


# mistral template


def test_mistral_format_single_user_prompt():
    mistral_prompt_template = PromptTemplate(
        dict(User="[INST] {MESSAGE} [/INST]", Assistant="{MESSAGE}</s> "),
        bos_token="<s>",
        eos_token="</s>",
        response_extract_pattern="[/INST]",
    )

    m21 = mistral_prompt_template.format_messages({"role": "User", "content": "Calculate the sum of 2 and 3."})
    m21_expected = "<s>[INST] Calculate the sum of 2 and 3. [/INST]"
    assert m21 == m21_expected, "mistral formatting single user prompt failed"


def test_mistral_format_single_message_chat():
    mistral_prompt_template = PromptTemplate(
        dict(User="[INST] {MESSAGE} [/INST]", Assistant="{MESSAGE}</s> "),
        bos_token="<s>",
        eos_token="</s>",
        response_extract_pattern="[/INST]",
    )
    m22 = mistral_prompt_template.format_messages(
        [
            {"role": "User", "content": "Calculate the sum of 2 and 3."},
            {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
        ]
    )
    m22_expected = """<s>[INST] Calculate the sum of 2 and 3. [/INST]The sum of 2 and 3 is 5.</s> """
    assert m22 == m22_expected, "mistral formatting single message chat failed"

    extract_m22 = mistral_prompt_template.extract_response(m22)
    extract_m22_expected = "The sum of 2 and 3 is 5."
    assert extract_m22 == extract_m22_expected, "mistral formatting single message chat extract failed"


def test_mistral_format_multi_message_chat():
    mistral_prompt_template = PromptTemplate(
        dict(User="[INST] {MESSAGE} [/INST]", Assistant="{MESSAGE}</s> "),
        bos_token="<s>",
        eos_token="</s>",
        response_extract_pattern="[/INST]",
    )
    m23 = mistral_prompt_template.format_messages(
        [
            {"role": "User", "content": "Calculate the sum of 2 and 3."},
            {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
            {"role": "User", "content": "Thank you! Could you also calculate the sum of 5 and 7?"},
        ]
    )
    m23_expected = "<s>[INST] Calculate the sum of 2 and 3. [/INST]The sum of 2 and 3 is 5.</s> [INST] Thank you! Could you also calculate the sum of 5 and 7? [/INST]"
    assert m23 == m23_expected, "mistral formatting multi message chat failed"


# UserAssistantPromptTemplate
def test_extra_id_user_assistant_format_single_user_prompt():
    extra_id_user_assistant_format = UserAssistantPromptTemplate(
        user_format="<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n",
        assistant_format="{MESSAGE}\n",
        system_format="<extra_id_0>System\n{MESSAGE}\n",
        system_default_message="",
        eos_token="<extra_id_1>",
        response_extract_pattern="<extra_id_1>Assistant\n",
    )

    m31 = extra_id_user_assistant_format.format_messages(
        {"role": UserAssistantPromptTemplate.Role.User, "content": "Calculate the sum of 2 and 3."}
    )
    m31_expected = """<extra_id_0>System

<extra_id_1>User
Calculate the sum of 2 and 3.
<extra_id_1>Assistant
"""
    assert m31 == m31_expected, "extra_id user assistant format single user prompt failed"


def test_extra_id_user_assistant_format_single_user_prompt_create_user_message():
    extra_id_user_assistant_format = UserAssistantPromptTemplate(
        user_format="<extra_id_1>User\n{MESSAGE}\n<extra_id_1>Assistant\n",
        assistant_format="{MESSAGE}\n",
        system_format="<extra_id_0>System\n{MESSAGE}\n",
        system_default_message="",
        eos_token="<extra_id_1>",
        response_extract_pattern="<extra_id_1>Assistant\n",
    )

    m32 = extra_id_user_assistant_format.format_messages(
        extra_id_user_assistant_format.create_user_message("Calculate the sum of 2 and 3.")
    )
    m32_expected = """<extra_id_0>System

<extra_id_1>User
Calculate the sum of 2 and 3.
<extra_id_1>Assistant
"""
    assert m32 == m32_expected, "extra_id user assistant format single user prompt create user message failed"


def test_mistral_user_assistant_format_single_user_prompt():
    mistral_user_assistant_format = UserAssistantPromptTemplate(
        user_format="[INST] {MESSAGE} [/INST]",
        assistant_format="{MESSAGE}</s> ",
        bos_token="<s>",
        eos_token="</s>",
        response_extract_pattern="[/INST]",
    )

    m41 = mistral_user_assistant_format.format_messages(
        {"role": UserAssistantPromptTemplate.Role.User, "content": "Calculate the sum of 2 and 3."}
    )
    m41_expected = "<s>[INST] Calculate the sum of 2 and 3. [/INST]"
    assert m41 == m41_expected, "mistral user assistant format single user prompt failed"


def test_mistral_user_assistant_format_single_user_prompt_create_user():
    mistral_user_assistant_format = UserAssistantPromptTemplate(
        user_format="[INST] {MESSAGE} [/INST]",
        assistant_format="{MESSAGE}</s> ",
        bos_token="<s>",
        eos_token="</s>",
        response_extract_pattern="[/INST]",
    )

    m42 = mistral_user_assistant_format.format_messages(
        mistral_user_assistant_format.create_user_message("Calculate the sum of 2 and 3.")
    )
    m42_expected = "<s>[INST] Calculate the sum of 2 and 3. [/INST]"
    assert m42 == m42_expected, "mistral user assistant format single user prompt create user failed"


def test_collated_single_chat_single_message():
    single_chat_single_message = [{"role": "User", "content": "Calculate the sum of 2 and 3."}]

    expected_collated_chat_messages = [{"role": ["User"], "content": ["Calculate the sum of 2 and 3."]}]

    collated_chat_messages = ChatTemplateHelper.collate_chat_messages(single_chat_single_message)
    assert collated_chat_messages == expected_collated_chat_messages, "collating single chat messages failed"

    collated_chat_messages = ChatTemplateHelper.collate_chat_messages([single_chat_single_message])
    assert (
        collated_chat_messages == expected_collated_chat_messages
    ), "collating single chat with single message failed"


def test_collated_single_chat():
    single_chat_messages = [
        {"role": "User", "content": "Calculate the sum of 2 and 3."},
        {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
        {"role": "User", "content": "Thank you! Could you also calculate the sum of 5 and 7?"},
    ]

    expected_collated_chat_messages = [
        {"role": ["User"], "content": ["Calculate the sum of 2 and 3."]},
        {"role": ["Assistant"], "content": ["The sum of 2 and 3 is 5."]},
        {"role": ["User"], "content": ["Thank you! Could you also calculate the sum of 5 and 7?"]},
    ]

    collated_chat_messages = ChatTemplateHelper.collate_chat_messages(single_chat_messages)
    assert collated_chat_messages == expected_collated_chat_messages, "collating single chat messages failed"

    collated_chat_messages = ChatTemplateHelper.collate_chat_messages([single_chat_messages])
    assert collated_chat_messages == expected_collated_chat_messages, "collating single chat messages failed"


def test_collated_multi_chat_messages():
    chat_1 = [
        {"role": "User", "content": "Calculate the sum of 2 and 3."},
        {"role": "Assistant", "content": "The sum of 2 and 3 is 5."},
        {"role": "User", "content": "Thank you! Could you also calculate the sum of 5 and 7?"},
    ]

    chat_2 = [
        {"role": "User", "content": "Calculate the sum of 56 and 21."},
        {"role": "Assistant", "content": "The sum of 56 and 21 is 77."},
        {"role": "User", "content": "Thank you! Could you also calculate the sum of 15 and 17?"},
    ]

    chat_list = [chat_1, chat_2]

    collated_chat_messages = ChatTemplateHelper.collate_chat_messages(chat_list)

    expected_collated_chat_messages = [
        {"role": ["User", "User"], "content": ["Calculate the sum of 2 and 3.", "Calculate the sum of 56 and 21."]},
        {"role": ["Assistant", "Assistant"], "content": ["The sum of 2 and 3 is 5.", "The sum of 56 and 21 is 77."]},
        {
            "role": ["User", "User"],
            "content": [
                "Thank you! Could you also calculate the sum of 5 and 7?",
                "Thank you! Could you also calculate the sum of 15 and 17?",
            ],
        },
    ]

    assert collated_chat_messages == expected_collated_chat_messages, "collating multi chat messages failed"


def test_collated_multi_chat_each_with_single_message():
    chat_1 = [{"role": "User", "content": "Calculate the sum of 2 and 3."}]

    chat_2 = [{"role": "User", "content": "Calculate the sum of 56 and 21."}]

    expected_collated_chat_messages = [
        {"role": ["User", "User"], "content": ["Calculate the sum of 2 and 3.", "Calculate the sum of 56 and 21."]},
    ]

    collated_chat_messages = ChatTemplateHelper.collate_chat_messages([chat_1, chat_2])

    assert collated_chat_messages == expected_collated_chat_messages, "collating multi chat with single message failed"
