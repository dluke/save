
# %%
import os
from glob import glob
from PIL import Image

def replace_color(bmp_image):
	color_to_replace = (255, 0, 255)

	# Get the data of the image
	data = bmp_image.getdata()
	
	# Create a new list to hold the updated image data
	new_data = []

	 # Replace the specified color with transparency
	for item in data:
		# Check if this pixel matches the color to replace
		if item[0] == color_to_replace[0] and item[1] == color_to_replace[1] and item[2] == color_to_replace[2]:
			# Replace with transparent
			new_data.append((0, 0, 0, 0))
		else:
			# Keep the original pixel
			new_data.append(item)
		
	bmp_image.putdata(new_data)

def convert_bmp_to_png(input_path, output_path):
	# Load the .bmp image
	bmp_image = Image.open(input_path).convert("RGBA")

	replace_color(bmp_image)
	
	# Create a new image with an alpha channel
	png_image = Image.new("RGBA", bmp_image.size)
	
	# Paste the .bmp image onto the new image
	png_image.paste(bmp_image, (0, 0), bmp_image)
	
	# Save the image as .png with an alpha channel
	png_image.save(output_path, "PNG")


# %%

if __name__	== "__main__":
	# operate on all .bmp in a folder
	dry_run = False
	delete = True

	for input_path in glob('./*.bmp'):
		name, ext = os.path.splitext(input_path)
		output_path = name + '.png'
		print("convert {} -> {}".format(input_path, output_path))
		if not dry_run:
			convert_bmp_to_png(input_path, output_path)
			if delete:
				os.remove(input_path)

