from tree_of_thoughts.treeofthoughts import TreeofThoughts, OptimizedTreeofThoughts
from tree_of_thoughts.openai_language_model import CustomLanguageModel, OptimizedOpenAILanguageModel, OpenAILanguageModel
from tree_of_thoughts.huggingface_language_model import HuggingFaceLanguageModel
use_v2 = False

# api_key="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"
# api_base= "" # leave it blank if you simply use default openai api url

# if not use_v2:
#     #v1
#     model = OpenAILanguageModel(api_key=api_key, api_base=api_base)
# else:
#     #v2 parallel execution, caching, adaptive temperature
#     model = OptimizedOpenAILanguageModel(api_key=api_key, api_base=api_base)

model = HuggingFaceLanguageModel(cookie_path="cookies.json")

#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

#cot or propose
strategy="cot"

# value or vote
evaluation_strategy = "value"

if not use_v2:
    #create an instance of the tree of thoughts class v1
    tree_of_thoughts = TreeofThoughts(model, search_algorithm)
else:
    #or v2 -> dynamic beam width -< adjust the beam width [b] dynamically based on the search depth quality of the generated thoughts
    tree_of_thoughts= OptimizedTreeofThoughts(model, search_algorithm)

input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"
k = 5
T = 3
b = 5
vth = 0.5

#call the solve method with the input problem and other params
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, )

#use the solution in your production environment
print(solution)
