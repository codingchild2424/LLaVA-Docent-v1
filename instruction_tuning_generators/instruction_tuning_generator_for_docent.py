import os
import json
import openai
from dotenv import dotenv_values, load_dotenv
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import multiprocessing
import time
import pandas as pd
import random
from tqdm import tqdm

config = dotenv_values("../.env")

openai.api_key = config.get('OPENAI_API_KEY')

src_path = "../datasets/llava_instruct_150k.json"
dst_path = "../datasets/llava_instruct_150k_docent_v1.json"

#####################################################
# GPT call
#####################################################
def gpt_call(
    prompt,
    model="gpt-4-1106-preview"
    ):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
                    {"role": "user", "content": prompt},
                ]
    )
    output_text = response["choices"][0]["message"]["content"]

    print("output_text", output_text)

    return output_text


def prompt_func(instruction_prompt):

    prompt_template = PromptTemplate(
        input_variables=[
            "prompt"
            ],
        template="\n".join([
            "data:{prompt}",
            "Change \"gpt\" value to docent style in art museum.",
            "The name of \"human\" and \"gpt\" must be maintained.",
            "data:"
        ])
    )
    prompt_template = prompt_template.format(
        prompt=instruction_prompt
    )
    print("prompt_template", prompt_template)

    return prompt_template


call_counter = 0  # This counter is for the gpt_call rate limit
line_counter = multiprocessing.Value('i', 0)  # Shared counter for lines


def append_to_dst(dst_path, data):
    with open(dst_path, "a") as f:
        json.dump(data, f)
        f.write(",\n")

def main():

    temp_dst_path = "/workspace/Coding/LLaVA/datasets/llava_docent_v2-91.json"

    prompt_template_path = "/workspace/Coding/LLaVA/instruction_tuning_generators/prompt_templates/prompt_template.txt"
    framework_path = "/workspace/Coding/LLaVA/instruction_tuning_generators/prompt_templates/framework_v3_Anderson.tsv"
    persona_path = "/workspace/Coding/LLaVA/instruction_tuning_generators/prompt_templates/persona.tsv"
    wiki_art_data_path = "/workspace/Coding/LLaVA/instruction_tuning_generators/prompt_templates/wiki_art_data.tsv"

    #prompt_template
    with open(prompt_template_path, "r") as f:
        prompt_template_text = f.read()

    # framework_df, persona_df, wiki_art_data_df
    framework_df = pd.read_csv(framework_path, sep="\t")
    persona_df = pd.read_csv(persona_path, sep="\t")
    wiki_art_data_df = pd.read_csv(wiki_art_data_path, sep="\t")

    prompt_template = PromptTemplate(
        input_variables=[
            "reaction",
            "perceptual_analysis_representation",
            "perceptual_analysis_formal_analysis",
            "perceptual_analysis_formal_characterization",
            "personal_interpretation",
            "contextual_examination",
            "synthesis_resolution",
            "synthesis_evaluation",
            "student_persona",
            "artwork_name",
            "artwork_explanation",
            "artist_name",
            "category",
            "date",
            "style",
            "media",
            ],
        template=prompt_template_text
    )
    """
    {
        "id": "000000033471",
        "image": "000000033471.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "from": "gpt",
                "value": "The artistic rendering of the bus presents it in hues of white and red."
            }
        ]
    }
    """

    for idx_1, i in tqdm(enumerate(range(2))):
        for idx_2, row in tqdm(enumerate(wiki_art_data_df.iterrows())):

            row_data = row[1]
            wiki_number = row_data["number"]

            if wiki_number is not 91:
                continue

            artwork_name = row_data["artwork_title"]
            artwork_explanation = row_data["explanation"]
            artist_name = row_data["artist"]
            category = row_data["category"]
            date = row_data["date"]
            style = row_data["style"]
            media = row_data["media"]

            reaction = framework_df["Reaction"][random.randint(0, framework_df["Reaction"].shape[0]-1)]
            perceptual_analysis_representation = framework_df["Perceptual Analysis_Representation"][random.randint(0, framework_df["Perceptual Analysis_Representation"].shape[0]-1)]
            perceptual_analysis_formal_analysis = framework_df["Perceptual Analysis_Formal Analysis"][random.randint(0, framework_df["Perceptual Analysis_Formal Analysis"].shape[0]-1)]
            perceptual_analysis_formal_characterization = framework_df["Perceptual Analysis_Formal Characterization"][random.randint(0, framework_df["Perceptual Analysis_Formal Characterization"].shape[0]-1)]
            personal_interpretation = framework_df["Personal Interpretation"][random.randint(0, framework_df["Personal Interpretation"].shape[0]-1)]
            contextual_examination = framework_df["Contextual Examination"][random.randint(0, framework_df["Contextual Examination"].shape[0]-1)]
            synthesis_resolution = framework_df["Synthesis_Resolution"][random.randint(0, framework_df["Synthesis_Resolution"].shape[0]-1)]
            synthesis_evaluation = framework_df["Synthesis_Evaluation"][random.randint(0, framework_df["Synthesis_Evaluation"].shape[0]-1)]

            student_persona = persona_df["persona_text"][random.randint(0, persona_df["persona_text"].shape[0]-1)]

            prompt_template = prompt_template.format(
                reaction=reaction,
                perceptual_analysis_representation=perceptual_analysis_representation,
                perceptual_analysis_formal_analysis=perceptual_analysis_formal_analysis,
                perceptual_analysis_formal_characterization=perceptual_analysis_formal_characterization,
                personal_interpretation=personal_interpretation,
                contextual_examination=contextual_examination,
                synthesis_resolution=synthesis_resolution,
                synthesis_evaluation=synthesis_evaluation,
                student_persona=student_persona,
                artwork_name=artwork_name,
                artwork_explanation=artwork_explanation,
                artist_name=artist_name,
                category=category,
                date=date,
                style=style,
                media=media,
            )

            gpt_response = gpt_call(prompt_template)

            gpt_response_list = gpt_response.split("\n")

            conversations = []

            for idx, one_turn in enumerate(gpt_response_list):
                if one_turn == "":
                    continue
                else:
                    if one_turn.split(":")[0] == "student":
                        if idx == 0:
                            conversations.append({"from": "human", "value": "<image>\n" + one_turn.split(":")[1]})
                        else:
                            conversations.append({"from": "human", "value": one_turn.split(":")[1]})
                    elif one_turn.split(":")[0] == "teacher":
                        if idx == 0:
                            conversations.append({"from": "gpt", "value": "<image>\n" + one_turn.split(":")[1]})
                        else:
                            conversations.append({"from": "gpt", "value": one_turn.split(":")[1]})
                    else:
                        continue

            # post processing
            post_result = {
                "id": str(wiki_number),
                "image": str(wiki_number) + ".jpg",
                "conversations": conversations
            }

            append_to_dst(temp_dst_path, post_result)
        

    # prompt_template = prompt_template.format(
    #     prompt=instruction_prompt
    # )


    # # Check if the dst_path exists
    # start_index = 0
    # if os.path.exists(dst_path):
    #     # Count processed data and set the start index to continue from there
    #     start_index = count_processed_data()

    # # Clear or initialize the destination file if starting from scratch
    # if start_index == 0:
    #     with open(dst_path, "w") as k:
    #         k.write("[\n")

    # with open(src_path, "r") as f:
    #     data = json.load(f)
    
    # # Skip already processed data
    # data = data[start_index:]

    # # Create chunks for multiprocessing
    # num_cores = multiprocessing.cpu_count()
    # chunk_size = len(data) // num_cores
    # chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # with multiprocessing.Pool(num_cores) as pool:
    #     # Use tqdm here to show progress bar
    #     results = list(tqdm(pool.imap(process_data, chunks), total=len(chunks)))

    # # Close the JSON array in the destination file
    # with open(dst_path, "rb+") as k:
    #     # Go to the second last character in the file
    #     k.seek(-2, 2)
    #     k.truncate()
    #     k.write(b"\n]")

if __name__ == '__main__':
    main()