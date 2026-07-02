from interactive_generation import InteractiveGeneration
import torch
from level_dataset import visualize_samples, convert_to_level_format, positive_negative_caption_split, append_absence_captions
from captions.caption_match import compare_captions, process_scene_segments, TOPIC_KEYWORDS
from create_ascii_captions import assign_caption
from captions.util import extract_tileset
from util.sampler import scene_to_ascii
import argparse
import util.common_settings as common_settings
from util.sampler import SampleOutput
from models.pipeline_loader import get_pipeline
from models.fdm_pipeline import FDMPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate levels using a trained diffusion model")
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--tileset", default=common_settings.MM2_TILESET, help="Descriptions of individual tile types")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--automatic_negative_captions", action="store_true", default=False, help="Automatically create negative captions for prompts so the user doesn't have to")
    parser.add_argument("--automatic_absence_captions", action="store_true", default=False, help="Automatically create absence captions for prompts so the user doesn't have to")
    parser.add_argument(
        "--game",
        type=str,
        default="Mario Maker",
        choices=["Mario Maker"],
        help="Which game to create a model for (affects sample style and tile count)"
    )

    return parser.parse_args()

class InteractiveLevelGeneration(InteractiveGeneration):
    def __init__(self, args):
        super().__init__(
            {
                "caption": str,
                "width": int,
                "negative_prompt": str,
                "start_seed": int,
                "end_seed": int,
                "num_inference_steps": int,
                "guidance_scale": float,
            },
            default_parameters={
                "width":  width,
                "start_seed": 1,
                "end_seed": 1,  # Will be set to start_seed if blank
                "num_inference_steps": common_settings.NUM_INFERENCE_STEPS,
                "guidance_scale": common_settings.GUIDANCE_SCALE,
                "caption": "",
                "negative_prompt": "",
            }
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = get_pipeline(args.model_path).to(self.device)

        if args.automatic_negative_captions or isinstance(self.pipe, FDMPipeline) or not self.pipe.supports_negative_prompt:
            # removed negative caption as an input
            self.input_parameters.pop('negative_prompt', None)
            self.default_parameters.pop('negative_prompt', None)

        if args.automatic_negative_captions and (isinstance(self.pipe, FDMPipeline) or not self.pipe.supports_negative_prompt):
            raise ValueError("Automatic negative caption generation is not possible with a model that doesn't support it")

        if args.tileset:
            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(args.tileset)

        self.args = args

        print(f"Tileset in use: {self.args.tileset}")

    def generate_image(self, param_values, generator, **extra_params):
        if self.args.automatic_negative_captions:
            pos, neg = positive_negative_caption_split(param_values["caption"], True)
            param_values["negative_prompt"] = neg

        if self.args.automatic_absence_captions:
            param_values["caption"] = append_absence_captions(
                param_values["caption"], TOPIC_KEYWORDS
            )
        try:
            images = self.pipe(
                generator=generator,
                **param_values
            ).images
            print(f"PARAM VALUES: ", param_values)
        except Exception as e:
            print(f"Error during image generation: {e}")
            return None

        # Convert to indices
        sample_tensor = images[0].unsqueeze(0)
        sample_indices = convert_to_level_format(sample_tensor)

        # Add level data to the list
        scene = sample_indices[0].tolist()

        actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, self.args.describe_absence)
        level_width = common_settings.MM2_WIDTH

        compare_score = compare_captions(param_values.get("caption", ""), actual_caption)
        print(f"Comparison score: {compare_score}")

        # Use the new function to process scene segments
        average_score, segment_captions, segment_scores = process_scene_segments(
            scene=scene,
            segment_width=level_width,
            prompt=param_values.get("caption", ""),
            id_to_char=self.id_to_char,
            char_to_id=self.char_to_id,
            tile_descriptors=self.tile_descriptors,
            describe_locations=False,
            describe_absence=self.args.describe_absence,
            verbose=True
        )

        # Ask if user wants to play level
        play_level = input("Do you want to play this level? (y/n): ").strip().lower()
        if play_level == 'y':
            print("Playing level...")
            # Mario Maker (MM2), so use the Python astar/ check instead of the Java sim
            from astar.astar_traversability_check import astar_console_report
            console_output = astar_console_report(scene, id_to_char=self.id_to_char,
                                                  tile_descriptors=self.tile_descriptors)
            print(console_output)
        elif play_level == 'n':
            print("Level not played.")
        else:
            print("Unknown input: Level not played.")

        samples = visualize_samples(images, game="MM2")

        return samples

    def get_extra_params(self, param_values):
        if "negative_prompt" in param_values and param_values["negative_prompt"] == "":
            del param_values["negative_prompt"]

        if param_values["caption"] == "":
            del param_values["caption"]

        param_values["output_type"] = "tensor"

        return dict()

if __name__ == "__main__":
    args = parse_args()

    height = common_settings.MM2_HEIGHT
    width = common_settings.MM2_WIDTH
    args.tile_size = common_settings.MM2_TILE_PIXEL_DIM
    args.tileset = common_settings.MM2_TILESET
    args.num_tiles = len(extract_tileset(args.tileset)[0])

    ig = InteractiveLevelGeneration(args)
    ig.start()
