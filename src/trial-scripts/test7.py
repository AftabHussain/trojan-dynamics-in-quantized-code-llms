import pandas as pd
import svgwrite
import argparse

# Function to convert probability to color (e.g., using a gradient)
def probability_to_color(prob):
    # Map the probability (0 to 1) to a color scale (blue to red)
    red = int(255 * prob)  # Higher probability = more red
    green = int(255 * (1 - prob))  # Lower probability = less green
    return f'#{red:02x}{green:02x}00'  # RGB format

# Main function to generate SVG
def generate_svg(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Create an SVG drawing
    dwg = svgwrite.Drawing('tokens.svg', profile='tiny', size=(800, 100))

    # Set initial x position
    x_position = 10

    last_row = 0

    # Loop through each token and add it to the SVG
    for index, row in df.iterrows():
        token = row['token']
        probability = row['output_probability']

        #print (len(df.loc[last_row]['token']))

        #if last_row>0:
        # Update x position for the next token (with some spacing)
        x_position += 50 + len(df.loc[last_row]['token'])*20  # Adjust spacing as needed

        # Get the color for the current token based on its probability
        color = probability_to_color(probability)

        # Add the token as text to the SVG
        dwg.add(dwg.text(token, insert=(x_position, 50), fill=color, font_size='20px', font_family='Arial'))

        last_row += 1

    # Save the SVG file
    dwg.save()

# Entry point for the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SVG from CSV tokens and probabilities.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    args = parser.parse_args()
    
    generate_svg(args.csv_file)

