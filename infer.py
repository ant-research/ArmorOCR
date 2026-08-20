from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "YOUR_MODEL_PATH", dtype="auto", device_map="auto"
)


processor = AutoProcessor.from_pretrained("YOUR_MODEL_PATH")


all_messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "examples/example_1.png"},
                {"type": "text", "text": "Please identify the hidden text in the image. Please put your reasoning process inside <analyze></analyze> and your final recognized text inside <answer></answer>."},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "examples/example_2.png"},
                {"type": "text", "text": "What is the faintly visible text in the central region of the pattern formed by dense vertical black lines? Please put your reasoning process inside <analyze></analyze> and your final recognized text inside <answer></answer>."},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "examples/example_3.png"},
                {"type": "text", "text": "What text appears beneath the dot pattern in gray? Please put your reasoning process inside <analyze></analyze> and your final recognized text inside <answer></answer>."},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "examples/example_4.jpeg"},
                {"type": "text", "text": "What is the white text above the QR code? Please put your reasoning process inside <analyze></analyze> and your final recognized text inside <answer></answer>."},
            ],
        }
    ],
]

for i, messages in enumerate(all_messages, 1):
    print("=" * 50)
    print(f"Example {i}")
    print("=" * 50)

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

