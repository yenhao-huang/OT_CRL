import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--config', type=str, required=True, help='configs')
    args = parser.parse_args()
    config = args.config
    # Define the base directory and the drampower command template
    base_dir = os.path.expanduser(f"~/ramulator/lib/energy_analysis/cmd/{config}")
    home_dir = "/home/user/ramulator-pim/common/DRAMPower"
    drampower_cmd_template = "{}/drampower -m {}/memspecs/MICRON_4Gb_DDR4-2400_8bit_A.xml -c {}"

    # List all files in the base directory
    files = os.listdir(base_dir)

    # Iterate over each file and run the drampower command
    for file in files:
        file_path = os.path.join(base_dir, file)
        if os.path.isfile(file_path):
            cmd = drampower_cmd_template.format(home_dir, home_dir, file_path)
            try:
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                print(f"Successfully executed command for {file}:\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Error executing command for {file}:\n{e.stderr}")

