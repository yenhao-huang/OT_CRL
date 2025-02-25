import argparse


def change_config(i_path, o_path, config):
    with open(i_path, 'r') as file:
        filedata = file.read()
        
    filedata = filedata.replace('{mapping}', config)
        
    with open(o_path, 'w') as file:
        file.write(filedata)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--mapping', type=str, required=True, help='Workload')
    args = parser.parse_args()
    
    i_path='/home/user/ramulator/configs/DDR3-sample.cfg'
    o_path='/home/user/ramulator/configs/DDR3-config.cfg'
    if args.mapping == "row_bank_col":
        config = "/home/user/ramulator/mappings/ours/row_bank_col.map"
    elif args.mapping == "bank_row_col":
        config = "/home/user/ramulator/mappings/ours/bank_row_col.map"
    elif args.mapping == "bit_rev":
        config = "/home/user/ramulator/mappings/ours/bit_reversal.map"
    elif args.mapping == "permutation":
        config = "/home/user/ramulator/mappings/ours/permutation.map"
    else:
        raise "NOT THIS MAPPING"
    
    change_config(i_path, o_path, config)