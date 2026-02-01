"""
Modal-based GPU LLM service for document classification.

Provides on-demand GPU access for LLM inference using open models.

Usage:
    # Deploy with default model (Phi-3)
    modal deploy modal_llm.py

    # Deploy with specific model
    OKB_LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct modal deploy modal_llm.py

Then call from Python:
    llm = modal.Cls.from_name("knowledge-llm", "LLM")()
    response = llm.complete.remote("Classify this document", system="You are a classifier")
"""

import os

import modal

app = modal.App("knowledge-llm")

# Model is set via environment variable at deploy time
# Default to Phi-3 which doesn't require HuggingFace approval
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
MODEL_ID = os.environ.get("OKB_LLM_MODEL", DEFAULT_MODEL)

# GPU type - L4 recommended for speed/cost balance
DEFAULT_GPU = "L4"
GPU_TYPE = os.environ.get("OKB_MODAL_GPU", DEFAULT_GPU)

# Container image with transformers and torch
llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.42.0",  # For quantization
        "hf_transfer",  # Fast downloads
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "OKB_LLM_MODEL": MODEL_ID,
        "OKB_MODAL_GPU": GPU_TYPE,
    })
)


@app.cls(
    image=llm_image,
    gpu=GPU_TYPE,
    timeout=300,
    scaledown_window=300,  # Keep warm for 5 min
    retries=1,
)
class LLM:
    """GPU-accelerated LLM for document classification."""

    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Read model from environment (set at deploy time)
        self.model_id = os.environ.get("OKB_LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
        print(f"Loading model: {self.model_id}")

        # Use 4-bit quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on {self.model.device}")

    @modal.method()
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> dict:
        """Generate a completion for the given prompt.

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Dict with 'content', 'model', 'input_tokens', 'output_tokens'
        """
        import torch

        # Build messages in chat format
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_tokens = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        output_tokens = outputs.shape[1] - input_tokens
        response_text = self.tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True,
        )

        return {
            "content": response_text.strip(),
            "model": self.model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    @modal.method()
    def complete_batch(
        self,
        prompts: list[str],
        system: str | None = None,
        max_tokens: int = 256,
    ) -> list[dict]:
        """Generate completions for multiple prompts.

        Args:
            prompts: List of user prompts
            system: Optional system prompt (same for all)
            max_tokens: Maximum tokens per response

        Returns:
            List of response dicts
        """
        # For now, process sequentially (batched inference is more complex)
        return [self.complete(prompt, system=system, max_tokens=max_tokens) for prompt in prompts]


# For testing
@app.local_entrypoint()
def test():
    """Test the LLM."""
    llm = LLM()

    system = "You are a document classifier. Respond with JSON: {action, reason}"
    prompt = """Classify this email:

Subject: 50% OFF - Limited Time Offer!
From: deals@marketing-spam.com

Don't miss out on our biggest sale of the year!
Click here to claim your discount before it expires!"""

    print("Testing LLM classification...")
    response = llm.complete.remote(prompt, system=system, max_tokens=100)

    print(f"Model: {response['model']}")
    print(f"Tokens: {response['input_tokens']} in, {response['output_tokens']} out")
    print(f"Response:\n{response['content']}")
