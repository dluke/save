# %%

import os
import shutil 
from glob import glob
from pathlib import Path

example = r"e:\Nicht arbeiten\Cheats for Games\Eador\grabber\Pics\Fairy dragon\Fairy_Dragon_ground.bmp"

file_list = os.listdir()
print(file_list)

# %%

# main
if __name__ == '__main__':

	dry_run = False

	os.getcwd()
	file_list = glob('*.png')
	for item in file_list:
		# simply_name
		name = Path(item.replace("\\", "/")).name
		print('rename {} -> {}'.format(item, name))

		if not dry_run:
			shutil.move(item, name)
