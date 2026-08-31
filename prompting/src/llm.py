import os
import json
from enum import Enum, auto
from dotenv import load_dotenv
from pathlib import Path
from anthropic import Anthropic

class GenerationMode(Enum):
    TRAIT_BASED = auto()
    LOOKUP = auto()

class ClaudeGenerator:
    """Handles interactions with Anthropic with support for Trait-Based and Rubric-Lookup modes."""
    
    def __init__(self, mode: GenerationMode, model="claude-sonnet-4-6"):
        load_dotenv()
        self.model = model
        self.mode = mode
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        self.score_cols = [
            'overall', 'content', 'prompt adherence', 'language', 'narrativity', 
            'organization', 'word choice', 'sentence fluency', 'conventions', 'style', 'voice'
        ]

        # Base paths
        base_path = Path(__file__).parent
        prompt_dir = base_path.parent / 'prompts'
        
        # 1. Load Trait Descriptions (used as default or in Trait-Based mode)
        with open(base_path / 'descriptions.json', 'r') as f:
            self.descriptions = json.load(f)

        # 2. Load Mode-Specific Templates and Data
        if self.mode == GenerationMode.TRAIT_BASED:
            template_name = 'user_prompt_template_hyp_mix.md'
            self.rubrics = {}
        else:
            template_name = 'user_prompt_template_db.md'
            rubric_path = base_path.parent / 'trait_descriptions.json'
            with open(rubric_path, 'r') as f:
                self.rubrics = json.load(f)

        with open(prompt_dir / template_name, 'r') as f:
            self.user_template = f.read()

        self.system_prompt = "You simulate student writing behavior conditioned on overall proficiency and trait-level profile."

    def format_user_prompt(self, context):
        """Routing function to call the correct formatting logic."""
        if self.mode == GenerationMode.LOOKUP:
            return self._format_lookup_prompt(context)
        elif self.mode == GenerationMode.TRAIT_BASED:
            return self._format_trait_prompt(context)

    def _format_trait_prompt(self, context):
        """Logic for standard trait-based descriptions."""
        scores = context['scores']
        ranges = context['ranges']
        prompt_id = str(context.get('prompt_id', ''))

        scores_section = ""
        characteristics_section = ""

        for trait in self.score_cols:
            score_val = scores.get(trait)
            range_val = ranges.get(trait)
            if score_val and score_val != "Not Applicable":
                scores_section += f"- {trait}: {score_val} {range_val}\n"
                
                # Use generic description
                if trait in self.descriptions and self.descriptions[trait]:
                    characteristics_section += f"**{trait.capitalize()}:** {self.descriptions[trait]}\n\n"

        return self._fill_template(scores_section, characteristics_section, prompt_id, context['instruction'])

    def _format_lookup_prompt(self, context):
        """Logic for specific rubric-level instruction lookups."""
        scores = context['scores']
        ranges = context['ranges']
        prompt_id = str(context.get('prompt_id', ''))
        # Standardize key to "prompt_1"
        rubric_key = f"prompt_{prompt_id}" if not prompt_id.startswith("prompt_") else prompt_id

        scores_section = ""
        characteristics_section = ""

        for trait in self.score_cols:
            score_val = scores.get(trait)
            range_val = ranges.get(trait)
            if score_val != "Not Applicable":
                scores_section += f"- {trait}: {score_val} {range_val}\n"
                
                if trait != "overall":
                    # Deep lookup: prompt_id -> trait -> score
                    trait_rubric = self.rubrics.get(rubric_key).get(trait)
                    specific_instruction = trait_rubric.get(str(score_val))

                    if specific_instruction:
                        characteristics_section += f"**{trait.capitalize()} {score_val} rubric:** {specific_instruction}\n\n"

        return self._fill_template(scores_section, characteristics_section, prompt_id, context['instruction'])

    def _fill_template(self, scores_section, characteristics_section, prompt_id, instruction):
        """Helper to perform the string replacements."""
        with open(Path(__file__).parent.parent / 'prompts' / f'prompt_{prompt_id}_examples.md', 'r') as f:
            examples = f.read()

        return (
            self.user_template
            .replace('{scores}', scores_section.strip())
            .replace('{prompt_id}', prompt_id)
            .replace('{characteristics}', characteristics_section.strip())
            .replace('{instruction}', instruction)
            .replace('{examples}', examples)
        )

    def generate_single(self, context):
        """Live API call for testing."""
        user_prompt = self.format_user_prompt(context)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return resp.content[0].text

    def prepare_batch_line(self, custom_id, context):
        """Formats a single line for the Anthropic Batch array."""
        user_prompt = self.format_user_prompt(context)
        return {
            "custom_id": str(custom_id),
            "params": {
                "model": self.model,
                "max_tokens": 1000,
                "system": self.system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            }
        }
        
    def submit_batch(self, requests):
        """Submits the list of requests to Anthropic."""
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id
        
    def check_batch_status(self, batch_id):
        """Returns the current status of the batch (e.g., 'processing', 'ended')."""
        return self.client.messages.batches.retrieve(batch_id)
        
    def get_batch_results(self, batch_id):
        """Streams the results back once the batch is ended."""
        return self.client.messages.batches.results(batch_id)