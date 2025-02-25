import random

def generate_stream_access_trace(num_accesses, output_file):
    start_address = 0x0
    address_increment = 0x40
    address_limit = 0xFFFFFFFF
    
    # Open the output file for writing
    with open(output_file, "w") as f:
        for i in range(num_accesses):
            address = start_address + i * address_increment
            if address > address_limit:
                raise "Address over limits"
            f.write(f"{i}:\tread\t0x{address:x}\n")

    print(f"Stream access trace written to {output_file}")
 
def generate_random_access_trace(num_accesses, output_file):  
    address_min = 0x40
    address_max = 0xFFFFFFFF

    # Open the output file for writing
    with open(output_file, "w") as f:
        for i in range(num_accesses):
            address = random.randint(address_min, address_max)
            f.write(f"{i}:\tread\t0x{address:x}\n")

    print(f"Random access trace written to {output_file}")
    
if __name__ == "__main__":
    num_accesses = int(1e9)
    o_f = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/resources/traces/ours/synthetic/stream_1e9.stl"
    generate_stream_access_trace(num_accesses=num_accesses, output_file=o_f)
    o_f = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/resources/traces/ours/synthetic/random_1e9.stl"
    generate_random_access_trace(num_accesses=num_accesses, output_file=o_f)