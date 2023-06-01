import open_gpt
from open_gpt.profile import end_measure, log_measures, start_measure, compute_module_sizes

start_measures = start_measure()
model = open_gpt.create_model(
    'RWKV/rwkv-raven-1b5', device='cpu', precision='fp16'
)
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model loading")
module_sizes = compute_module_sizes(model)

PROMPTS = [
    "Tell me about ravens.",
    "Write a song about ravens.",
    "Explain the following metaphor: Life is like cats.",
    "Generate a list of adjectives that describe a person as brave.",
    "You have $100, and your goal is to turn that into as much money as possible with AI and Machine Learning. Please respond with detailed plan.",
]

start_measures = start_measure()
for prompt in PROMPTS:
    output = model.generate(
        prompt,
        max_length=100,
        temperature=1.2,
        top_k=0,
        top_p=0.5,
        repetition_penalty=0.4,
        do_sample=True,
        num_return_sequences=1,
    )
    print(output)

end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")