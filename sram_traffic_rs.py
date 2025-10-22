import math
import csv 
from tqdm import tqdm

class EnergyCounter:
    """Energy consumption counter for various operations"""
    def __init__(self):
        # Counter for different operations
        self.dram_read_count = 0
        self.dram_write_count = 0
        self.sram_read_count = 0
        self.sram_write_count = 0
        self.pe_data_transfer_count = 0  # Data transfer between neighboring PEs
        self.rf_read_count = 0
        self.rf_write_count = 0
        self.alu_operations_count = 0
        
        # Energy values per operation (임시 값들, 나중에 수정 가능)
        self.energy_per_dram_read = 500.0    # pJ
        self.energy_per_dram_write = 500.0   # pJ
        self.energy_per_sram_read = 10.0      # pJ
        self.energy_per_sram_write = 10.0     # pJ
        self.energy_per_pe_transfer = 3.0    # pJ
        self.energy_per_rf_read = 1.0        # pJ
        self.energy_per_rf_write = 1.0       # pJ
        self.energy_per_alu_op = 1.0         # pJ
        
    def increment_dram_read(self, count=1):
        self.dram_read_count += count
        
    def increment_dram_write(self, count=1):
        self.dram_write_count += count
        
    def increment_sram_read(self, count=1):
        self.sram_read_count += count
        
    def increment_sram_write(self, count=1):
        self.sram_write_count += count
        
    def increment_pe_data_transfer(self, count=1):
        self.pe_data_transfer_count += count
        
    def increment_rf_read(self, count=1):
        self.rf_read_count += count
        
    def increment_rf_write(self, count=1):
        self.rf_write_count += count
        
    def increment_alu_operations(self, count=1):
        self.alu_operations_count += count
    
    def calculate_total_energy(self):
        """Calculate total energy consumption"""
        total_energy = (
            self.dram_read_count * self.energy_per_dram_read +
            self.dram_write_count * self.energy_per_dram_write +
            self.sram_read_count * self.energy_per_sram_read +
            self.sram_write_count * self.energy_per_sram_write +
            self.pe_data_transfer_count * self.energy_per_pe_transfer +
            self.rf_read_count * self.energy_per_rf_read +
            self.rf_write_count * self.energy_per_rf_write +
            self.alu_operations_count * self.energy_per_alu_op
        )
        return total_energy
    
    def print_energy_breakdown(self):
        """Print detailed energy breakdown"""
        print("="*60)
        print("ENERGY CONSUMPTION BREAKDOWN")
        print("="*60)
        print(f"DRAM Read Operations:     {self.dram_read_count:>10} x {self.energy_per_dram_read:>6.1f} pJ = {self.dram_read_count * self.energy_per_dram_read:>10.1f} pJ")
        print(f"DRAM Write Operations:    {self.dram_write_count:>10} x {self.energy_per_dram_write:>6.1f} pJ = {self.dram_write_count * self.energy_per_dram_write:>10.1f} pJ")
        print(f"SRAM Read Operations:     {self.sram_read_count:>10} x {self.energy_per_sram_read:>6.1f} pJ = {self.sram_read_count * self.energy_per_sram_read:>10.1f} pJ")
        print(f"SRAM Write Operations:    {self.sram_write_count:>10} x {self.energy_per_sram_write:>6.1f} pJ = {self.sram_write_count * self.energy_per_sram_write:>10.1f} pJ")
        print(f"PE Data Transfer:         {self.pe_data_transfer_count:>10} x {self.energy_per_pe_transfer:>6.1f} pJ = {self.pe_data_transfer_count * self.energy_per_pe_transfer:>10.1f} pJ")
        print(f"RF Read Operations:       {self.rf_read_count:>10} x {self.energy_per_rf_read:>6.1f} pJ = {self.rf_read_count * self.energy_per_rf_read:>10.1f} pJ")
        print(f"RF Write Operations:      {self.rf_write_count:>10} x {self.energy_per_rf_write:>6.1f} pJ = {self.rf_write_count * self.energy_per_rf_write:>10.1f} pJ")
        print(f"ALU Operations:           {self.alu_operations_count:>10} x {self.energy_per_alu_op:>6.1f} pJ = {self.alu_operations_count * self.energy_per_alu_op:>10.1f} pJ")
        print("-"*60)
        print(f"TOTAL ENERGY:             {self.calculate_total_energy():>35.1f} pJ")
        print("="*60)
        
    def save_to_csv(self, filename="energy_breakdown.csv"):
        """Save energy breakdown to CSV file"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Component', 'Operation_Count', 'Energy_per_Op_pJ', 'Total_Energy_pJ'])
            writer.writerow(['DRAM_Read', self.dram_read_count, self.energy_per_dram_read, 
                           self.dram_read_count * self.energy_per_dram_read])
            writer.writerow(['DRAM_Write', self.dram_write_count, self.energy_per_dram_write, 
                           self.dram_write_count * self.energy_per_dram_write])
            writer.writerow(['SRAM_Read', self.sram_read_count, self.energy_per_sram_read, 
                           self.sram_read_count * self.energy_per_sram_read])
            writer.writerow(['SRAM_Write', self.sram_write_count, self.energy_per_sram_write, 
                           self.sram_write_count * self.energy_per_sram_write])
            writer.writerow(['PE_Data_Transfer', self.pe_data_transfer_count, self.energy_per_pe_transfer, 
                           self.pe_data_transfer_count * self.energy_per_pe_transfer])
            writer.writerow(['RF_Read', self.rf_read_count, self.energy_per_rf_read, 
                           self.rf_read_count * self.energy_per_rf_read])
            writer.writerow(['RF_Write', self.rf_write_count, self.energy_per_rf_write, 
                           self.rf_write_count * self.energy_per_rf_write])
            writer.writerow(['ALU_Operations', self.alu_operations_count, self.energy_per_alu_op, 
                           self.alu_operations_count * self.energy_per_alu_op])
            writer.writerow(['TOTAL', '', '', self.calculate_total_energy()])

# In Row Stationary, there's single SRAM (scratchpad mem)
# In each 'compute_2d_convolution', there may be several folds.
# Then 'ifmap reuse' is followed, so SRAM always contains one ifmap data.
# But under unlimited RF size, it can happen 'channel accumulation' in PE.

# Before one 'compute_2d_conv()', SRAM must be prepared one filter's one channel data at least.
# In summary, in each fold, SRAM holds
# 1) One ifmap data (pinned for whole 3D conv)
# 2) one filter

# Address
# Filter Address : (filt_idx, row, col, chan_idx)
# Ifmap Address : (row, col, chan_idx)
# Ofmap Address : (filt_idx, row, col)

def get_1d_addr_filt(curr_filt,
                    filt_h, curr_row,
                    filt_w, curr_col,
                    num_chan, curr_chan):
    return ((curr_filt * filt_h + curr_row) * filt_w + curr_col) * num_chan + curr_chan

def get_1d_addr_ifmap(curr_row, ifmap_w, curr_col, num_chan, curr_chan):
    return (curr_row * ifmap_w + curr_col) * num_chan + curr_chan

def get_1d_addr_ofmap(curr_filt, 
                      ofmap_h, curr_row, ofmap_w):
    return (curr_filt * ofmap_h + curr_row) * ofmap_w + 0


def sram_traffic(
        arr_h=12,                # systolic array num rows (default: Eyeriss)
        arr_w=14,                # systolic array num cols (default: Eyeriss)
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3,
        strides=1, num_filt=8,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv",
        rf_ifmap_wsz = 12, rf_filt_wsz = 224, rf_psum_wsz = 24,
        ifmap_sram_wsz = 108*512, filt_sram_wsz = 108*512, ofmap_sram_wsz = 108*512
    ):

    # Initialize energy counter
    energy_counter = EnergyCounter()

    # Parameters
    ofmap_h = math.floor((ifmap_h - filt_h + strides) / strides)
    ofmap_w = math.floor((ifmap_w - filt_w + strides) / strides)

    # 1D-Conv Primitives (prim)
    # 'each primitive operates on one row of filter weights and
    #  one row of ifmap pixels, and generates one row of psums.'

    # Fold Parameters
    # Ofmap height fold is triggered when ofmap_h > arr_w
    # Ofmap width fold is triggered when ofmap_w > PE's psum spad sz
    # Channel fold is triggered when ifmap & filt spad size is small
    # Filter group can be made when there's remained space in PE's filt spad
    num_oh_fold = int(math.ceil(ofmap_h / arr_w))
    num_ow_fold = int(math.ceil(ofmap_w / rf_psum_wsz))
    cnt_ch_per_fold = calc_cnt_ch_per_fold(rf_ifmap_wsz, rf_filt_wsz, filt_w)
    num_c_fold = int(math.ceil(num_channels / cnt_ch_per_fold))
    #num_filt_group = int(math.ceil(rf_filt_wsz / (cnt_ch_per_fold * filt_w)))

    # Utility Parameters
    cycle = 0
    prev_cycle = 0
    util = 0
    compute_cycles = 0
    lmcol_cycle = 0

    # Systolic array use-state Parameters
    use_arr_h = min(arr_h, filt_h)
    use_arr_w = min(arr_w, ofmap_h)
    use_psum_spad = rf_psum_wsz

    # Progress bar
    tot = num_channels
    pbar = tqdm(total=tot)

    # Parallel Window -> Making Filter IDX List
    filt_parallel = int(math.floor(arr_h / filt_h))
    num_fgroup = int(math.ceil(num_filt / filt_parallel))
    fgr_list = []
    '''
    for cfgr in range(num_fgroup):
        fgr = []
        for filt in range(filt_parallel):
            fgr.append(cfgr * filt_parallel + filt)
        fgr_list.append(fgr)
    '''
    for cfgr in range(num_fgroup):
        fgr = []
        for filt in range(filt_parallel):
            if (cfgr * filt_parallel + filt) < num_filt:
                fgr.append(cfgr * filt_parallel + filt)
        fgr_list.append(fgr)

    # REQUEST
    for chn in range(num_channels):
        orow_start = 0
        use_arr_w = min(arr_w, ofmap_h)
        for oh_fold in range(num_oh_fold):
            if (oh_fold == num_oh_fold - 1) and (ofmap_h % arr_w > 0):
                use_arr_w = ofmap_h % arr_w
        
            orow_needs = use_arr_w
            orow_list = make_seq_list(orow_start, orow_needs)

            ocol_start = 0
            use_psum_spad = rf_psum_wsz
            for ow_fold in range(num_ow_fold):
                if (ow_fold == num_ow_fold - 1) and (ofmap_w % rf_psum_wsz > 0):
                    use_psum_spad = ofmap_w % rf_psum_wsz
                
                ocol_needs = use_psum_spad
                ocol_list = make_seq_list(ocol_start, ocol_needs)

                irow_needs = use_arr_h + (use_arr_w - 1) * strides
                irow_start = orow_start * strides
                irow_list = make_irow_list(irow_start, irow_needs, strides, use_arr_h)
                
                icol_needs = filt_w + (ocol_needs - 1) * strides
                icol_start = ocol_start * strides
                icol_list = make_seq_list(icol_start, icol_needs)

                frow_list = make_seq_list(0, use_arr_h)
                fcol_list = make_seq_list(0, filt_w)

                for fgr in fgr_list:
                    # Ifmap data is only requested at first time
                    if fgr[0] == 0:
                        cycle = gen_read_trace(cycle, ofmap_h, ofmap_w,
                                            ifmap_w, filt_h, filt_w, num_channels,
                                            irow_list, icol_list,
                                            frow_list, fcol_list,
                                            orow_list, ocol_list,
                                            chn, fgr,
                                            ifmap_base, filt_base, ofmap_base,
                                            sram_read_trace_file,
                                            energy_counter)
                    else:
                        cycle = gen_read_trace(cycle, ofmap_h, ofmap_w,
                                            ifmap_w, filt_h, filt_w, num_channels,
                                            [], [], # Empty Ifmap request
                                            frow_list, fcol_list,
                                            orow_list, ocol_list,
                                            chn, fgr,
                                            ifmap_base, filt_base, ofmap_base,
                                            sram_read_trace_file,
                                            energy_counter)

                    """
                    if orow_start == 0 and ocol_start == 0:
                        cycle = gen_read_trace(cycle, ofmap_h, ofmap_w,
                                            ifmap_w, filt_h, filt_w, num_channels,
                                            irow_list, icol_list,
                                            frow_list, fcol_list,
                                            orow_list, ocol_list,
                                            chn, curr_filt,
                                            ifmap_base, filt_base, ofmap_base,
                                            sram_read_trace_file)
                    else:
                        cycle = gen_read_trace(cycle, ofmap_h, ofmap_w,
                                            ifmap_w, filt_h, filt_w, num_channels,
                                            irow_list, icol_list,
                                            [], [],
                                            orow_list, ocol_list,
                                            chn, curr_filt,
                                            ifmap_base, filt_base, ofmap_base,
                                            sram_read_trace_file)
                    """
                    
                    # Stall cycle needs since previous psum should be positioned in psum spad completely.
                    cycle += use_psum_spad

                    lmcol_cycle = compute_lmcol_cycle(cycle,
                                                        use_arr_h, use_arr_w,
                                                        filt_w, use_psum_spad)

                    cycle = gen_write_trace(lmcol_cycle, fgr,
                                            ofmap_h, ofmap_w,
                                            orow_list, ocol_list,
                                            ofmap_base, sram_write_trace_file)


                    # Update energy counters
                    if energy_counter:
                        # RF read for weight (for each PE)
                        energy_counter.increment_rf_read((use_arr_h * len(fgr) * use_arr_w) * filt_w * ocol_needs) 
                        # RF read for ifmap (for each PE)
                        energy_counter.increment_rf_read((use_arr_h * len(fgr) * use_arr_w) * filt_w * ocol_needs)   
                        # RF read for psum (for each PE)
                        energy_counter.increment_rf_read((use_arr_h * len(fgr) * use_arr_w) * filt_w * ocol_needs) 
                        # ALU ops
                        energy_counter.increment_alu_operations((use_arr_h * len(fgr) * use_arr_w) * filt_w * ocol_needs)
                        # RF write for psum (for each PE)
                        energy_counter.increment_rf_write((use_arr_h * len(fgr) * use_arr_w) * filt_w * ocol_needs)


                        # Energy for psum accumulation
                        # PE data transfer 
                        energy_counter.increment_pe_data_transfer(len(fgr) * (use_arr_h - 1) * use_arr_w)
                        # RF read for psum stored in PE (for psum accumulation)
                        energy_counter.increment_rf_read(len(fgr) * (use_arr_h - 1) * use_arr_w) 
                        # ALU ops
                        energy_counter.increment_alu_operations(len(fgr) * (use_arr_h - 1) * use_arr_w)


                        # SRAM write for psum (result of accumulation of psums)
                        energy_counter.increment_sram_write(len(fgr) * use_arr_w * ocol_needs)
                        # SRAM read for previous psum (for psum accumulation)
                        if chn > 0:
                            # energy_counter.increment_sram_read(len(fgr) * use_arr_w)
                            # ALU ops for psum writeback
                            energy_counter.increment_alu_operations(len(fgr) * use_arr_w * ocol_needs)                     


                    # Utils
                    delta_cycle = cycle - prev_cycle
                    util_this_fold: float = ((use_arr_h * len(fgr)) * use_arr_w) / (arr_h * arr_w)

                    compute_cycles += delta_cycle
                    util += int(util_this_fold * delta_cycle)
                    prev_cycle = cycle

                ocol_start += ocol_needs
            
            orow_start += orow_needs

        pbar.update(1)    

    # Final Utils
    final = str(cycle)
    final_util = (util / compute_cycles) * 100
    return (final, final_util, energy_counter)
            


# --- Calc & Trace functions ---

def gen_read_trace(
        cycle = 0, ofmap_h = 5, ofmap_w = 5,
        ifmap_w = 7, filt_h = 3, filt_w = 3, num_channels = 3,
        irow_list = [], icol_list = [],
        frow_list = [], fcol_list = [],
        orow_list = [], ocol_list = [],
        chn = 0, fgroup = [],
        ifmap_base = 0, filt_base = 1000000, ofmap_base = 2000000,
        sram_read_trace_file = "sram_read.csv",
        energy_counter = None
):
    outfile = open(sram_read_trace_file, 'a')

    # Left Column, Upper Row PE부터 순차로 시작
    # -> 같은 Ifmap row를 사용하는 PE는 같은 pixel을 request

    sram_reads_cnt = 0

    # Ifmap req address
    ifmap_entry_list = []
    if len(irow_list) > 0:
        col_idxs = [0 for _ in range(len(irow_list))]
        irs = 1

        while col_idxs[-1] < len(icol_list):
            entry = ""

            for ir in range(min(irs, len(irow_list))):
                if col_idxs[ir] >= len(icol_list):
                    continue

                entry += str(ifmap_base + 
                            (irow_list[ir] * ifmap_w + icol_list[col_idxs[ir]]) 
                            * num_channels + chn) + ", "
                
                sram_reads_cnt += 1

                col_idxs[ir] += 1
            
            ifmap_entry_list.insert(0, entry)
            irs += 1

    # Filter req address
    filt_entry_list = []
    if len(frow_list) > 0:
        for fc in fcol_list:
            entry = ""
            for filt_idx in fgroup:
                for fr in frow_list:
                    entry += str(filt_base + ((filt_idx * filt_h + fr) * filt_w + fc) *
                                num_channels + chn) + ", "

                    sram_reads_cnt += 1

            filt_entry_list.insert(0, entry)
    
    # Psum req address
    psum_entry_list = []
    if len(orow_list) > 0:
        for ocol in ocol_list:
            entry = ""
            for filt_idx in fgroup:
                for orow in orow_list:
                    entry += str(ofmap_base + (filt_idx * ofmap_h + orow) * ofmap_w + ocol) + ", "

                    sram_reads_cnt += 1

            psum_entry_list.insert(0, entry)


    # Write entry
    for cyc in range(max(len(ifmap_entry_list), max(len(psum_entry_list), len(filt_entry_list)))):
        entry = str(cycle) + ", "

        if len(ifmap_entry_list) > 0:
            entry += ifmap_entry_list.pop()
            sram_reads_cnt += 1
        if len(filt_entry_list) > 0:
            entry += filt_entry_list.pop()
            sram_reads_cnt += 1
        if len(psum_entry_list) > 0:
            entry += psum_entry_list.pop()
            sram_reads_cnt += 1

        entry += "\n"
        outfile.write(entry)
        cycle += 1


    outfile.close()
    return cycle


def gen_write_trace(
        cycle = 0, fgroup = [],
        ofmap_h = 5, ofmap_w = 5,
        orow_list = [], ocol_list = [],
        ofmap_base = 2000000,
        sram_write_trace_file = "sram_write.csv"
):
    outfile = open(sram_write_trace_file, 'a')

    # Don't need to count SRAM write count here, it's counted in gen_sram_traffic()

    # Since each column finishes sequentially
    for orow in orow_list:
        entry = str(cycle) + ", "
        for filt_idx in fgroup:
            for ocol in ocol_list:
                entry += str(ofmap_base + (filt_idx * ofmap_h + orow) * ofmap_w + ocol) + ", "
        entry += "\n"
        outfile.write(entry)
        
        cycle += 1

    outfile.close()

    return cycle
   

def compute_lmcol_cycle(
        cycle = 0,
        use_arr_h = 12, use_arr_w = 14,
        filt_w = 3, use_psum_spad = 24
):
    # Current cycle : Rightmost Bottom PE's data read is finished
    # Leftmost Bottom PE's last data read is finished at
    lmbot_cycle = cycle - (use_arr_w - 1)

    # 1) Calc primitive cycle [PE 내부] 
    # -> lmbot_cycle은 이미 마지막 데이터 읽은 것이므로, prim연산은 끝남
    # prim_cycle = use_psum_spad * chan_needs

    # > Psum Accumulation
    # - Since each PE started sequentially,
    # - at the moment that bottom PE finished its primitive,
    # - The acc is still being done in upper area
    # => So, there only takes 'psum width' cycles for final
    lmbot_cycle += use_psum_spad * filt_w

    return lmbot_cycle


def calc_cnt_ch_per_fold(
        rf_ifmap_wsz = 12, rf_filt_wsz = 224,
        filt_w = 3
):
    max_ifmap_ch = int(rf_ifmap_wsz / filt_w)
    max_filt_ch = int(rf_filt_wsz / filt_w)

    return min(max_ifmap_ch, max_filt_ch)


def make_seq_list(
        start = 0, needs = 1
):
    seq_list = list()
    for i in range(needs):
        seq_list.append(start + i)
    
    return seq_list


def make_irow_list(
        irow_start = 0, irow_needs = 1, strides = 1, use_arr_h = 12
):
    # 251010 todo : stride로 인해 불연속일 경우가 있는지 확인해야 함
    irow_list = list()
    for i in range(irow_needs):
        irow_list.append(irow_start + i)
    
    return irow_list