from pathlib import Path

def assign_bits(start_bit):
    total_bits = 26
    Ro_bits = 15
    Co_bits = 7
    Ba_bits = 2
    Bg_bits = 2
    
    lines = []
    if Ro_bits + start_bit > 26:
        raise "no this mapping"
    
    # Calculate Ro range
    Ro_start = start_bit
    Ro_end = start_bit + Ro_bits - 1
    lines.append(f"Ro {Ro_bits - 1}:0 = {Ro_end}:{Ro_start}\n")
    
    # Initialize remaining bits
    remaining_bits = list(range(total_bits))
    
    # Remove Ro bits from the remaining bits
    for bit in range(Ro_start, Ro_end + 1):
        remaining_bits.remove(bit)
    
    # Assign Co, Ba, and Bg bits
    Co_assign = remaining_bits[:Co_bits]
    Ba_assign = remaining_bits[Co_bits:Co_bits + Ba_bits]
    Bg_assign = remaining_bits[Co_bits + Ba_bits:Co_bits + Ba_bits + Bg_bits]
    
    # Format the assignments
    for i, co in enumerate(Co_assign):
        lines.append(f"Co {i} = {co}\n")
    for i, ba in enumerate(Ba_assign):
        lines.append(f"Ba {i} = {ba}\n")
    for i, bg in enumerate(Bg_assign):
        lines.append(f"Bg {i} = {bg}\n")
        
    return lines

HOME_DIR = "/home/user/ramulator/"
if __name__ == "__main__":
    config = 9
    lines = assign_bits(config)
    o_path = Path(HOME_DIR) / f"mappings/ours/ddr4/generate.map"
    
    with o_path.open("w") as f:
        f.writelines(lines)