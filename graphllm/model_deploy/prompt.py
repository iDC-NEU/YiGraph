from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
# from litellm import completion
# from utils.llm_env import OllamaEnv

llama_keyword_extract_prompt = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "Note, result should be in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
    # "Only response the results, do not say any word or explain.\n"
    "---------------------\n"
    "question: {question}\n"
    "---------------------\n"
    "KEYWORDS: ")
llama_keyword_extract_prompt_template = PromptTemplate(
    llama_keyword_extract_prompt,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)

llama_synonym_expand_prompt = (
    "Generate synonyms or possible form of keywords up to {max_keywords} in total,"
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms of keywords in comma-separated format: 'SYNONYMS: <synonyms>'\n"
    # "Note, result should be in one-line with only one 'SYNONYMS: ' prefix\n"
    # "Note, result should be in the following comma-separated format: 'SYNONYMS: <synonyms>\n"
    # "Only response the results, do not say any word or explain.\n"
    "Note, result should be in one-line, only response the results, do not say any word or explain.\n"
    "---------------------\n"
    "KEYWORDS: {question}\n"
    "---------------------\n"
    "SYNONYMS: ")

llama_synonym_expand_prompt_template = PromptTemplate(
    llama_synonym_expand_prompt,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)

command_keyword_extract_prompt = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "Note, result should be in the following comma-separated format, and start with KEYWORDS:'\n"
    # "Only response the results, do not say any word or explain.\n"
    "---------------------\n"
    "question: {question}\n"
    "---------------------\n"
    # "KEYWORDS: "
)
command_keyword_extract_prompt_template = PromptTemplate(
    command_keyword_extract_prompt,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)

command_synonym_expand_prompt = (
    "Generate synonyms or possible form of keywords up to {max_keywords} in total,"
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms of keywords in comma-separated format: 'SYNONYMS: <synonyms>'\n"
    # "Note, result should be in one-line with only one 'SYNONYMS: ' prefix\n"
    # "Note, result should be in the following comma-separated format: 'SYNONYMS: <synonyms>\n"
    # "Only response the results, do not say any word or explain.\n"
    "Note, result should be in one-line, only response the results, do not say any word or explain.\n"
    "---------------------\n"
    "KEYWORDS: {question}\n"
    "---------------------\n"
    # "SYNONYMS: "
)

command_synonym_expand_prompt_template = PromptTemplate(
    command_synonym_expand_prompt,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)

# "Note, result should be in the following comma-separated format: 'SYNONYMS: <synonyms>, only response the results, do not say any word or explain.\n"

gemma_synonym_expand_prompt = (
    # "<start_of_turn>user\n"
    "Generate synonyms or possible form of keywords up to {max_keywords} in total,"
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms of keywords in comma-separated format: 'SYNONYMS: <synonyms>'\n"
    # "Note, result should be in one-line with only one 'SYNONYMS: ' prefix\n"
    # "Note, result should be in the following comma-separated format: 'SYNONYMS: <synonyms>\n"
    # "Only response the results, do not say any word or explain.\n"
    "Note, result should be in one-line, only response the results, do not say any word or explain.\n"
    "<start_of_turn>user\n"
    "---------------------\n"
    "KEYWORDS: {question}\n"
    "---------------------\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"

    # "---------------------\n"
    # "KEYWORDS: {question}\n"
    # "---------------------\n"
    # "SYNONYMS: \n"
    # "<end_of_turn>\n"
    # "<start_of_turn>model\n"
)

gemma_synonym_expand_prompt_template = PromptTemplate(
    gemma_synonym_expand_prompt,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)

# <start_of_turn>user
# What is a good place for travel in the US?<end_of_turn>
# <start_of_turn>model
# California.<end_of_turn>
# <start_of_turn>user
# What can I do in California?<end_of_turn>
# <start_of_turn>model

# synonmys: ['Rory mcilroy', 'New venture', 'Tiger woods', 'Venture', 'Rory', 'New', 'Mcilroy', 'Woods', 'Tiger']

# KEYWORDS: Release Date, PlayStation, Release, PlayStation VR2, Headset, Sony, Date, VR2
# SYNONYMS: Release Day, PS, Launch, PSVR2, HMD, Sony Corp., Launch Date, Virtual Reality Headset 2

gemma_extract_prompt = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "Note, result should be in the following comma-separated format: 'KEYWORDS: <keywords>\n"
    # # "Only response the results, do not say any word or explain.\n"
    # "<start_of_turn>user\n"
    # "---------------------\n"
    # "question: What is the new venture launched by Tiger Woods and Rory McIlroy?\n"
    # "---------------------\n"
    # "<end_of_turn>\n"
    # "<start_of_turn>model\n"
    # "KEYWORDS: Rory McIlroy, new venture, Tiger Woods, venture, Rory, new, McIlroy, Woods, Tiger\n"
    # "<end_of_turn>\n"
    "<start_of_turn>user\n"
    "---------------------\n"
    "question: {question}\n"
    "---------------------\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
    # "KEYWORDS: \n"
    # "<end_of_turn>\n"
    # "<start_of_turn>model\n"
)

gemma_keyword_extract_prompt_template = PromptTemplate(
    gemma_extract_prompt,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


# def kg_prompt():

#     text = "old king cole was a merry old soul, and a merry old soul was he; He called for his pipe, and he called for his bowl, And he called for his fiddlers three. Every fiddler he had a fiddle, And a very fine fiddle had he; Oh, there's none so rare, as can compare, With King Cole and his fiddlers three."

#     messages = [{
#         "role": "user",
#         "content": f"{text}"
#     }, {
#         "role":
#         "system",
#         "content":
#         """
#                     You are an AI expert specializing in knowledge graph creation with the goal of capturing relationships based on a given input or request.
#                     Based on the user input in various forms such as paragraph, email, text files, and more.
#                     Your task is to create a knowledge graph based on the input.
#                     Nodes must have a label parameter. where the label is a direct word or phrase from the input.
#                     Edges must also have a label parameter, where the label is a direct word or phrase from the input.
#                     Response only with JSON in a format where we can jsonify in python and feed directly into  cy.add(data); to display a graph on the front-end.
#                     Make sure the target and source of edges match an existing node.
#                     Do not include the markdown triple quotes above and below the JSON, jump straight into it with a curly bracket.
#                     """
#     }]

#     response = completion(model="ollama/llama3:8b",
#                           messages=[{
#                               "content": "respond in 20 words. who are you?",
#                               "role": "user"
#                           }],
#                           api_base="http://localhost:11435")

#     print(response)


if __name__ == '__main__':
    # llm_env = OllamaEnv(llm_mode_name=args.llm, port=args.port)
    # llm_env = OllamaEnv(llm_mode_name='llama3:8b', 11435)

    kg_prompt()
