def read_DRAM():
    data = 1
    return data

def cache_isfull():
    return True

def get_replace_block():
    addr = 0x886
    return addr

def main():
    if access == "miss":
        if operation == "read":
            # read DRAM
            new_block = read_DRAM(addr)
            # update cache
            if cache_isfull():
                repl_block = get_replace_block()
                write_DRAM(repl_block)
            
            update_cache(new_block)
            return data
        elif operation == "write":
            new_block = read_DRAM(addr)
            if cache_isfull:
                repl_block = get_replace_block()
                write_DRAM(repl_block)
            update_cache(new_block)
            write_cache(data, addr)
            return "Done"
    elif access == "hit":
        if operation == "read":
            data = read_cache(addr)
            return data
        elif operation =="write":
            if is_second_write(addr):
                block = get_cur_block(addr)
                write_DRAM(block)
            write_cache(data, addr)
            return "Done"
                    