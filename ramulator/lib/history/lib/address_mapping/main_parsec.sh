HOME_DIR=/home/user/ramulator/
config_idx=0
python3 ${HOME_DIR}lib/change_config.py --mapping bank_row_col
python3 ${HOME_DIR}lib/address_mapping/main_ours_parsec.py --config_idx $config_idx

config_idx=1
python3 ${HOME_DIR}lib/change_config.py --mapping row_bank_col
python3 ${HOME_DIR}lib/address_mapping/main_ours_parsec.py --config_idx $config_idx

config_idx=2
python3 ${HOME_DIR}lib/change_config.py --mapping permutation
python3 ${HOME_DIR}lib/address_mapping/main_ours_parsec.py --config_idx $config_idx