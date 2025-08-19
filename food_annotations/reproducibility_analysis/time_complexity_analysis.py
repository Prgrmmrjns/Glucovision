import base64
import pandas as pd
from mistralai import Mistral
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import os
import time
import numpy as np
import random
import glob
from datetime import datetime

pd.options.mode.chained_assignment = None

# Load environment variables
load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)
model = "pixtral-large-latest"

class MacronutrientEstimations(BaseModel):
    simple_sugars: int
    complex_sugars: int
    proteins: int
    fats: int
    dietary_fibers: int
    explanation: str

macronutrients_instruction = f'''Examine the provided meal image to analyze and estimate its nutritional content accurately. 
Determine the amounts of simple sugars (like industrial sugar and honey), complex sugars (such as starch and whole grains), proteins, fats, and dietary fibers (found in fruits and vegetables), all in grams.
To assist in accurately gauging the scale of the meal, a 1 Swiss Franc coin, which has a diameter of 23 mm, may be present in the picture. 
Provide your assessment of each nutritional component in grams. All estimates should be given as a single whole number. 
Use the size of this coin as a reference to estimate the size of the meal and the amounts of the nutrients more precisely. 
If there is no coin in the picture or the meal is covered partially, estimate anyways.

Here are three examples of how to analyze different types of foods:

Example 1 - Pizza:
{MacronutrientEstimations(
    simple_sugars=6,
    complex_sugars=65,
    proteins=28,
    fats=32,
    dietary_fibers=5,
    explanation="This appears to be a medium-sized cheese pizza (about 12 inches in diameter). The crust provides most of the complex sugars (approximately 65g from flour). The tomato sauce contains some simple sugars (6g). The cheese provides most of the protein (28g) and fats (32g). The small amount of tomato sauce and potentially some vegetables contribute to the dietary fiber content (5g)."
)}

Example 2 - Bread with cheese and salad:
{MacronutrientEstimations(
    simple_sugars=4,
    complex_sugars=30,
    proteins=18,
    fats=15,
    dietary_fibers=8,
    explanation="The meal consists of a slice of bread with cheese and a side salad. The bread provides most of the complex sugars (30g from wheat flour). The cheese contributes protein (10g) and fats (10g). The salad provides dietary fibers (8g) and a small amount of simple sugars (4g from vegetables). The bread also contributes some protein (8g) and fats (5g)."
)}

Example 3 - Bottle of orange juice:
{MacronutrientEstimations(
    simple_sugars=42,
    complex_sugars=2,
    proteins=2,
    fats=0,
    dietary_fibers=1,
    explanation="This appears to be a standard 500ml bottle of orange juice. Orange juice is primarily composed of simple sugars (42g from natural fruit sugars). It contains minimal complex sugars (2g), a small amount of protein (2g), negligible fat (0g), and a small amount of dietary fiber (1g) from pulp residue in the juice."
)}

Format your response in the following JSON format:
{MacronutrientEstimations(
    simple_sugars=0,
    complex_sugars=0,
    proteins=0,
    fats=0,
    dietary_fibers=0,
    explanation=''
)}'''

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode('utf-8')

def make_mistral_api_call(image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": macronutrients_instruction
                },
                {
                    "type": "image_url",
                    "image_url": image_path,
                    "response_format": MacronutrientEstimations
                }
            ]
        }
    ]

    start_time = time.time()
    chat_response = client.chat.parse(
        model=model,
        messages=messages,
        response_format=MacronutrientEstimations,
        max_tokens=1024,
        temperature=0
    )
    end_time = time.time()
    
    response_json = chat_response.choices[0].message.content
    response = json.loads(response_json.strip('`json\n'))
    
    # Extract token usage
    usage = chat_response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    
    return response, end_time - start_time, prompt_tokens, completion_tokens, total_tokens

def get_all_images():
    """Get all available food images from the dataset"""
    all_images = []
    for patient in ['001', '002', '004', '006', '007', '008']:
        patient_path = f"../../diabetes_subset_pictures-glucose-food-insulin/{patient}/food_pictures/"
        if os.path.exists(patient_path):
            images = glob.glob(f"{patient_path}/*.jpg")
            for img in images:
                all_images.append((patient, os.path.basename(img), img))
    return all_images

def main():
    print("Starting time complexity analysis for mLLM macronutrient estimation...")
    
    # Get all available images
    all_images = get_all_images()
    print(f"Found {len(all_images)} total images across all patients")
    
    # Select 10 random images
    random.seed(42)  # For reproducibility
    selected_images = random.sample(all_images, min(10, len(all_images)))
    
    print(f"Selected {len(selected_images)} random images for analysis:")
    for patient, img_name, img_path in selected_images:
        print(f"  - Patient {patient}: {img_name}")
    
    # Analyze each image
    results = []
    
    for i, (patient, img_name, img_path) in enumerate(selected_images):
        print(f"\nProcessing image {i+1}/{len(selected_images)}: {patient}/{img_name}")
        
        try:
            base64_image = encode_image(img_path)
            response, elapsed_time, prompt_tokens, completion_tokens, total_tokens = make_mistral_api_call(base64_image)
            
            # Calculate total macronutrients
            total_macros = (response['simple_sugars'] + response['complex_sugars'] + 
                          response['proteins'] + response['fats'] + response['dietary_fibers'])
            
            result = {
                'patient': patient,
                'image': img_name,
                'elapsed_time_seconds': elapsed_time,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'simple_sugars': response['simple_sugars'],
                'complex_sugars': response['complex_sugars'],
                'proteins': response['proteins'],
                'fats': response['fats'],
                'dietary_fibers': response['dietary_fibers'],
                'total_macronutrients': total_macros,
                'explanation': response['explanation']
            }
            results.append(result)
            
            print(f"  ✓ Completed in {elapsed_time:.2f}s, {total_tokens} tokens")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary_stats = {
        'total_images_processed': len(results),
        'mean_time_seconds': results_df['elapsed_time_seconds'].mean(),
        'std_time_seconds': results_df['elapsed_time_seconds'].std(),
        'min_time_seconds': results_df['elapsed_time_seconds'].min(),
        'max_time_seconds': results_df['elapsed_time_seconds'].max(),
        'mean_prompt_tokens': results_df['prompt_tokens'].mean(),
        'mean_completion_tokens': results_df['completion_tokens'].mean(),
        'mean_total_tokens': results_df['total_tokens'].mean(),
        'std_total_tokens': results_df['total_tokens'].std(),
        'mean_total_macronutrients': results_df['total_macronutrients'].mean(),
        'std_total_macronutrients': results_df['total_macronutrients'].std()
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("TIME COMPLEXITY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Images processed: {summary_stats['total_images_processed']}")
    print(f"Mean processing time: {summary_stats['mean_time_seconds']:.2f} ± {summary_stats['std_time_seconds']:.2f} seconds")
    print(f"Time range: {summary_stats['min_time_seconds']:.2f} - {summary_stats['max_time_seconds']:.2f} seconds")
    print(f"Mean prompt tokens: {summary_stats['mean_prompt_tokens']:.1f}")
    print(f"Mean completion tokens: {summary_stats['mean_completion_tokens']:.1f}")
    print(f"Mean total tokens: {summary_stats['mean_total_tokens']:.1f} ± {summary_stats['std_total_tokens']:.1f}")
    print(f"Mean total macronutrients: {summary_stats['mean_total_macronutrients']:.1f} ± {summary_stats['std_total_macronutrients']:.1f} g")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'time_complexity_results_{timestamp}.csv', index=False)
    
    # Generate LaTeX summary
    latex_summary = f"""\\subsection{{Time Complexity Analysis of mLLM-based Macronutrient Estimations}}
\\label{{sec:time_complexity}}

To assess the computational requirements and practical feasibility of deploying mLLM-based macronutrient estimation in clinical settings, we conducted a time complexity analysis using 10 randomly selected meal images from the D1namo dataset. Each image was processed once using the Pixtral Large model with identical prompts and parameters.

The analysis revealed that macronutrient estimation requires an average of {summary_stats['mean_time_seconds']:.1f} ± {summary_stats['std_time_seconds']:.1f} seconds per image, with processing times ranging from {summary_stats['min_time_seconds']:.1f} to {summary_stats['max_time_seconds']:.1f} seconds. Token usage averaged {summary_stats['mean_total_tokens']:.0f} ± {summary_stats['std_total_tokens']:.0f} tokens per estimation, with {summary_stats['mean_prompt_tokens']:.0f} prompt tokens and {summary_stats['mean_completion_tokens']:.0f} completion tokens.

These computational requirements demonstrate that mLLM-based macronutrient estimation is feasible for real-time clinical applications, with processing times well under the typical 5-minute window for postprandial glucose prediction. The consistent token usage patterns suggest predictable computational costs, enabling efficient resource allocation in clinical deployment scenarios."""
    
    with open(f'time_complexity_latex_{timestamp}.txt', 'w') as f:
        f.write(latex_summary)
    
    print(f"\nResults saved to:")
    print(f"  - time_complexity_results_{timestamp}.csv")
    print(f"  - time_complexity_latex_{timestamp}.txt")
    
    return results_df, summary_stats

if __name__ == "__main__":
    results_df, summary_stats = main()
