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


def sram_traffic(
        dimension_rows=4,
        dimension_cols=4,
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3,
        strides=1, num_filt=8,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv"
    ):

    # Initialize energy counter
    energy_counter = EnergyCounter()

    # Dimensions of output feature map channel
    E_h = math.floor((ifmap_h - filt_h + strides) / strides)
    E_w = math.floor((ifmap_w - filt_w + strides) / strides)
    
    # Number of pixels in one convolution window
    px_per_conv_window = filt_h * filt_w * num_channels
    r2c = px_per_conv_window

    # Total number of ofmap px across all channels
    num_ofmap_px = E_h * E_w * num_filt
    e2  = E_h * E_w
    e2m = num_ofmap_px
    
    # Variables to calculate folds in runtime
    num_h_fold = 1
    num_v_fold = 1 
    max_parallel_window = 1

    # Variables for utilization calculation
    util = 0
    compute_cycles = 0

    if dimension_rows < px_per_conv_window:
        num_h_fold = math.ceil(px_per_conv_window/dimension_rows)
    else:
        max_parallel_window = math.floor(dimension_rows/ px_per_conv_window)

    reqd_cols = num_filt                    # Total number of cols to be mapped
    max_cols_per_v_fold = max_parallel_window * dimension_cols
    num_v_folds = math.ceil(reqd_cols / max_cols_per_v_fold)
    
    remaining_cols = reqd_cols
    cycles = 0
    prev_cycl = 0

    #print("Vertical folds = " +str(num_v_folds))
   
    # These are the starting addresses of filter weights in the memory 
    all_col_addr_list = []
    for c in range(num_filt):
        addr = (c) * r2c + filt_base 
        all_col_addr_list.append(addr)

    # These are the starting addresses of ifmap windows in the memory
    hc = ifmap_w * num_channels
    all_ifmap_base_addr = []
    for px in range(int(e2)):         #number of ofmap px in a ofmap channel
        addr = (px / E_w) * strides * hc + (px%E_w) * strides
        all_ifmap_base_addr.append(addr)

    for v in tqdm(range(int(num_v_folds))):
        #print("V fold id: " + str(v))
            
        # Take a slice of the starting addresses that are relevant for this v_fold 
        cols_this_fold = min(remaining_cols, max_parallel_window * dimension_cols)
        idx_start = v * dimension_cols
        idx_end = idx_start + cols_this_fold
        col_addr_list = all_col_addr_list[idx_start:idx_end]

        if num_h_fold > 1 :
           
            rem_h = r2c                     # Tracks the elements processed within a conv filter 
            next_ifmap_addr = ifmap_base    # Starts from the top left corner of the IFMAP matrix

            for h in range(num_h_fold):
                rows_this_fold = min(rem_h, dimension_rows) 
                #print("h fold id: " + str(h))

                # Values returned
                # cycles        -> Cycle count for the next operation ie. cycles elapsed + 1
                # col_addr_list -> The starting filter address for the next iteration
                cycles, col_addr_list   = gen_trace_filter_partial(
                                            col_addrs   = col_addr_list,
                                            cycle       = cycles,
                                            num_rows    = dimension_rows,
                                            remaining   = rows_this_fold,
                                            sram_read_trace_file = sram_read_trace_file,
                                            energy_counter = energy_counter
                                            )
                #print("Weights loaded by " + str(cycles) + " cycles")
                data_out_cycles     = cycles    #Store this cycle for parallel readout
                cycles_ifmap            = gen_trace_ifmap_partial(
                                            cycle = cycles,
                                            num_rows = dimension_rows, num_cols = dimension_cols,
                                            num_filters = num_filt,
                                            remaining = rem_h,
                                            remaining_filters = remaining_cols, 
                                            ifmap_h = ifmap_h, ifmap_w = ifmap_w,
                                            filt_h = filt_h, filt_w = filt_w,
                                            num_channels = num_channels,
                                            stride = strides, ifmap_base = ifmap_base,
                                            sram_read_trace_file = sram_read_trace_file,
                                            energy_counter = energy_counter
                                            )
                cycles_ofmap        = gen_trace_ofmap(
                                            cycle = data_out_cycles,
                                            num_rows = dimension_rows,
                                            num_cols = dimension_cols,
                                            ofmap_base = ofmap_base,
                                            window_size= rows_this_fold,
                                            parallel_window =1,
                                            num_ofmap_px = int(e2),
                                            filters_done = (v * dimension_cols),
                                            num_filter = num_filt,
                                            sram_write_trace_file = sram_write_trace_file,
                                            energy_counter = energy_counter
                                            ) 

                #print("IFMAPS processed by " + str(cycles) + " cycles")
                util_this_fold = (rows_this_fold * cols_this_fold) /(dimension_rows * dimension_cols)

                rem_h -= rows_this_fold
                cycles = max(cycles_ifmap, cycles_ofmap)

                del_cycl = cycles - prev_cycl
                util += util_this_fold *  del_cycl
                compute_cycles += del_cycl
                prev_cycl = cycles

        else:
            #filters_this_fold = min(remaining_cols, max_cols_per_v_fold)
            filt_done = v * max_parallel_window * dimension_cols
            rem = num_filt - filt_done

            parallel_window = math.ceil(rem / dimension_cols)
            parallel_window = int(min(max_parallel_window, parallel_window))
        
            cycles_filter = gen_filter_trace(
                                cycle = cycles,
                                num_rows = dimension_rows, num_cols = dimension_cols,
                                filt_h = filt_h, filt_w = filt_w, num_channels = num_channels,
                                col_addr = col_addr_list, 
                                parallel_window=parallel_window,
                                filters_this_fold=cols_this_fold,
                                sram_read_trace_file=sram_read_trace_file,
                                energy_counter=energy_counter
                                )

            cycles_ifmap, rows_this_fold\
                            = gen_ifmap_trace(
                            cycle = cycles_filter,
                            num_rows = dimension_rows, num_cols = dimension_cols,
                            ifmap_h = ifmap_h, ifmap_w = ifmap_w,
                            filt_h = filt_h, filt_w = filt_w,
                            num_channels = num_channels, stride = strides,
                            parallel_window = parallel_window,
                            sram_read_trace_file = sram_read_trace_file,
                            energy_counter = energy_counter
                            )

            cycles_ofmap = gen_trace_ofmap(
                            cycle = cycles_filter,
                            num_rows = dimension_rows, num_cols = dimension_cols,
                            ofmap_base = ofmap_base, 
                            parallel_window = parallel_window,
                            window_size = r2c,
                            num_ofmap_px = int(e2),
                            filters_done = int(v * max_parallel_window * dimension_cols),
                            num_filter = num_filt,
                            sram_write_trace_file = sram_write_trace_file,
                            energy_counter = energy_counter
                            )
            cycles = max(cycles_ifmap, cycles_ofmap)
            del_cycl = cycles - prev_cycl

            # Since multiple filters are being mapped on a single col due to large number of rows
            # util calculation is a little involved,
            # cols_this_fold --> number of filters mapped this fold
            rem = cols_this_fold
            tmp_util = 0
            for _ in range(parallel_window):
                col_used = min(rem, dimension_cols)
                row_used = r2c                      # Number of row used will always be in multiple of r2c,
                                                    # parallel window calc took care of this
                tmp_util += row_used * col_used
                rem -= col_used

            #util_this_fold = (rows_this_fold * cols_this_fold) /(dimension_rows * dimension_cols)
            util_this_fold = tmp_util /(dimension_rows * dimension_cols)
            util += util_this_fold * del_cycl
            compute_cycles += del_cycl
            prev_cycl = cycles

        remaining_cols -= cols_this_fold

    final = str(cycles)
    final_util = (util / compute_cycles) * 100
    #print("Compute finished at: " + str(final) + " cycles")
    return (final, final_util, energy_counter)


def gen_filter_trace(
        cycle = 0,
        num_rows = 4, num_cols = 4,
        filt_h = 3, filt_w = 3, num_channels = 3,
        col_addr = [],
        parallel_window = 1,
        filters_this_fold = 4,
        sram_read_trace_file = "sram_read.csv",
        energy_counter = None
):
    outfile = open(sram_read_trace_file,'a')
 
    # There is no data from the left side till the weights are fed in
    # This prefix is to mark the blanks
    prefix  = ""
    for r in range(num_rows):
        prefix += ", "

    # Calculate the convolution window size
    r2c = filt_h * filt_w * num_channels 

    rem = filters_this_fold                 # Track the number of filters yet to process

    #For each wrap around
    for w in range(parallel_window):
        # Number of active columns in this wrap
        cols = min(num_cols, rem)
        rem -= cols

        # For each row in the window
        for r in range(r2c):
            entry = str(cycle) + ", " + prefix
            cycle += 1
            
            # Count for this cycle
            cycle_sram_reads = 0
            cycle_rf_writes = 0
            
            # In each cycle, for each column feed one weight
            for c in range(cols):
                indx  = w * num_cols + c
                entry += str(col_addr[indx]) + ", "         
                col_addr[indx] += 1
                
                # WS: Filter weight loaded from SRAM to RF
                cycle_sram_reads += 1
                # cycle_rf_writes += 1  # Filter stored in RF

            if cols < num_cols:
                for _ in range(c, num_cols):
                    entry += ", "
            
            # Update energy counters
            if energy_counter:
                energy_counter.increment_sram_read(cycle_sram_reads)
                energy_counter.increment_rf_write(cycle_rf_writes)

            entry += "\n"
            outfile.write(entry)
 
    outfile.close()
    return cycle


def gen_ifmap_trace(
        cycle = 0,
        num_rows = 4, num_cols = 4,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w = 3,
        num_channels = 3, stride = 1,
        parallel_window = 1,
        sram_read_trace_file = "sram_read.csv",
        energy_counter = None
):
    outfile = open(sram_read_trace_file,'a')
    postfix = ""
    for c in range(num_cols):
        postfix += ", "
    
    E_h = math.floor((ifmap_h - filt_h + stride) / stride)
    E_w = math.floor((ifmap_w - filt_w + stride) / stride)
    e2  = E_h * E_w
    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels

    idle = num_rows - (r2c * parallel_window)
    idle = max(idle, 0)
    used_rows = num_rows - idle

    # Adding entries for columns and empty rows
    #print("Idle lanes = " + str(idle))
    idle += num_cols
    for i in range(idle):
        postfix += ", "
    postfix += "\n"

    base_addr = 0
    
    for e in range(int(e2)):
        entry = str(cycle) + ", "
        cycle += 1    

        # Count for this cycle
        cycle_sram_reads = 0
        cycle_rf_reads = 0  
        cycle_alu_ops = 0
        cycle_pe_transfers = 0

        #print("Cycle= " + str(cycle))
        #Inner loop for all the rows in array
        num_rows = r2c 
        row_entry = []
        for r in range(num_rows):
            row_idx = math.floor(r / rc)  # math.floor to get in integral value
            col_idx = r % rc 
            add = base_addr + row_idx * hc + col_idx 
            #print("Row idx " + str(row_idx) + " col_idx " + str(col_idx) +" add " + str(add))
            row_entry.append(add)

        # Reverse the printing order
        # Reversal is needed because the filter are stored in upside down order in the array
        # ie. last row has the first weight element
        l = len(row_entry)
        #print("Parallel windows = " + str(parallel_window))
        for w in range(parallel_window):
            #print("Window = " + str(w))
            for ridx in range(l):
                entry += str(row_entry[l - ridx -1]) + ", "
                # WS: IFMAP data read from SRAM 
                cycle_sram_reads += 1
        
        # WS: MAC operations with stationary filters
        current_active_cols = min(e + 1, num_cols)

        active_pes = used_rows * current_active_cols
        if active_pes > 0:
            cycle_rf_reads = active_pes     # Read filter weights
            cycle_alu_ops = active_pes      # MAC operations
            cycle_pe_transfers = used_rows * (current_active_cols - 1)  # PE-to-PE transfers
        
        # Update energy counters
        if energy_counter:
            energy_counter.increment_sram_read(cycle_sram_reads)
            energy_counter.increment_rf_read(cycle_rf_reads)
            energy_counter.increment_alu_operations(cycle_alu_ops)
            energy_counter.increment_pe_data_transfer(cycle_pe_transfers)

        entry += postfix
        outfile.write(entry)

        # Calculate the IFMAP addresses for next cycle
        px_this_row = (e+1) % E_w
        if px_this_row == 0:
            #print("New row")
            ifmap_row = math.floor(base_addr / hc)
            base_addr = (ifmap_row +  stride) * hc
        else:
            base_addr += stride * num_channels
        #print("OFAMP px = " + str(e+1) + " base_addr: " + str(base_addr))

    outfile.close()
    return cycle, used_rows


def gen_trace_filter_partial(
                    col_addrs=[],       #Ensure that this takes care of the v_folding
                    cycle=0,
                    num_rows=4,
                    remaining=4,
                    sram_read_trace_file="sram_read.csv",
                    energy_counter=None
):
        outfile = open(sram_read_trace_file, 'a')
        num_cols = len(col_addrs)

        # output formatting: Add empty commas for row addresses as no element is fed from the left
        prefix = ""
        for r in range(num_rows):
            prefix += ", "

        # Entries per cycle 
        for r in range(remaining):              # number of rows this cycle
            entry = str(cycle) + ", " + prefix

            # Count for this cycle
            cycle_sram_reads = 0
            cycle_rf_writes = 0

            for c in range(num_cols):
                entry += str(col_addrs[c]) + ", "
                col_addrs[c] += 1
                
                # WS: Filter weight loaded from SRAM to RF
                cycle_sram_reads += 1
                cycle_rf_writes += 1
            
            # Update energy counters
            if energy_counter:
                energy_counter.increment_sram_read(cycle_sram_reads)
                energy_counter.increment_rf_write(cycle_rf_writes)
            
            cycle += 1
            entry += "\n"
            outfile.write(entry)

        outfile.close()

        return cycle, col_addrs 


def gen_trace_ifmap_partial(
                    cycle = 0,
                    num_rows = 4, num_cols = 4,
                    remaining=4,
                    num_filters = 8,            #   
                    remaining_filters = 0,      # These two are used to track the reads of PS
                    ifmap_h = 4, ifmap_w = 4,
                    filt_h = 3, filt_w = 3,
                    num_channels = 3,
                    stride = 1, 
                    ifmap_base = 0, ofmap_base = 2000000,
                    sram_read_trace_file = "sram_read.csv",
                    energy_counter = None
):
    outfile = open(sram_read_trace_file, 'a')
    postfix = ""
    for c in range(num_cols):
        postfix += ", "
    postfix += "\n"

    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels
    E_w = (ifmap_w - filt_w + stride) / stride 
    E_h = (ifmap_h - filt_h + stride) / stride 

    num_ofmap_px = E_h * E_w
    index = r2c - remaining
    base_addr = 0 
            
    filter_done = num_filters - remaining_filters
    #outfile.write(str(filter_done) + ", " + str(num_filters)+", "+str(remaining_filters)+", "+ "\n")
    #ofmap_offset = filter_done * num_ofmap_px
    ofmap_offset = filter_done
    effective_cols = min(remaining_filters, num_cols)
    tick = 0                                # Proxy for clock to track input skewing

    # Outerloop for all ofmap pixels in an ofmap channel
    for e in range(int(num_ofmap_px)):
        entry = str(cycle) + ", "
        cycle += 1    

        # Count for this cycle
        cycle_sram_reads = 0
        cycle_rf_reads = 0
        cycle_alu_ops = 0
        cycle_pe_transfers = 0

        #print("Cycle= " + str(cycle))
        #Inner loop for all the rows in array
        num_rows = min(num_rows, remaining)
        row_entry = []
        for r in range(num_rows):
            row_idx = math.floor((index+r) / rc)  # math.floor to get in integral value
            col_idx = (index+r) % rc 
            add = base_addr + row_idx * hc + col_idx 
            #print("Row idx " + str(row_idx) + " col_idx " + str(col_idx) +" add " + str(add))
            row_entry.append(add)

        # Reverse the printing order
        # Reversal is needed because the filter are stored in upside down order in the array
        # ie. last row has the first weight element
        l = len(row_entry)
        for ridx in range(l):
            entry += str(row_entry[l - ridx -1]) + ", "
            # WS: IFMAP data read from SRAM
            cycle_sram_reads += 1
        
        # WS: MAC operations with stationary filters
        active_pes = num_rows * effective_cols
        if active_pes > 0:
            cycle_rf_reads = active_pes     # Read filter weights
            cycle_alu_ops = active_pes      # MAC operations
            cycle_pe_transfers = num_rows * max(effective_cols - 1, 0)  # PE-to-PE transfers
        
        # Update energy counters
        if energy_counter:
            energy_counter.increment_sram_read(cycle_sram_reads)
            energy_counter.increment_rf_read(cycle_rf_reads)
            energy_counter.increment_alu_operations(cycle_alu_ops)
            energy_counter.increment_pe_data_transfer(cycle_pe_transfers)

        # In case of partial mapping
        # index > 0 implies that there is a partial sum generated from prev h_fold
        # This partial sum is now fed from the top to be summed with the PS generated in this h_fold
        # The following part print the read addresses for PS
        # Anand : TODO, Implementation choice, do not support right now
        '''
        if index > 0:
            postfix = ""
            for c in range(effective_cols):
                if (tick - c) > -1:                       # Track PS reads for skew
                    a = (e - c) * num_filters + c        # e - c: Taking care of skew by c cycles
                    a = a + ofmap_base + ofmap_offset
                    postfix += str(a) + ", "
                else:
                    postfix += ", "
            tick += 1
            #print("Tick =", str(tick) + "Postfix= " + postfix)
            postfix += "\n"
        '''
        entry += postfix
        outfile.write(entry)

        px_this_row = (e+1) % E_w
        if px_this_row == 0:
            #print("New row")
            ifmap_row = math.floor(base_addr / hc)
            base_addr = (ifmap_row + stride) * hc
        else:
            base_addr += stride * num_channels
        #print("OFAMP px = " + str(e+1) + " base_addr: " + str(base_addr))

    outfile.close()
    return cycle


def gen_trace_ofmap(
                    cycle = 0,
                    num_rows = 4, num_cols =4,
                    ofmap_base = 2000000,
                    parallel_window = 1,
                    window_size = 27,
                    num_ofmap_px = 16,      # This is per ofmap channel
                    filters_done = 0,       # To track v fold
                    num_filter   = 8,       # To track if all filters have finished
                    sram_write_trace_file = "sram_write.csv",
                    energy_counter = None
):
    outfile = open(sram_write_trace_file,'a')
    #cycle = num_cols + cycle     # Accounts for the time taken to reduce accross all cols

    # Corner case when parallel_window = 1, but num_filter < num_cols
    if parallel_window > 1:
        cycle += num_cols
        cycle += window_size                # window_size == r2c
    else:
        rem    = (num_filter - filters_done)
        cycle += min(rem, num_cols)
        cycle += window_size

    #ofmap_add_offset  = filters_done * num_ofmap_px
    ofmap_add_offset  = filters_done
    remaining_filters = num_filter - filters_done
    
    effective_cols    = num_cols * parallel_window
    effective_cols    = min(effective_cols, remaining_filters)

    for e in range(int(num_ofmap_px)):
        entry = str(cycle) + ", "
        cycle += 1
        
        # Count for this cycle
        cycle_sram_writes = 0
        cycle_rf_reads = 0
        cycle_pe_transfers = 0
        
        done = filters_done
        for col in range(effective_cols):
            if done < num_filter:
                a = e * num_filter + col                # z first row major
                a = a + ofmap_add_offset + ofmap_base
                entry += str(a) + ", "
                
                # WS: OFMAP write to SRAM
                cycle_sram_writes += 1
                # WS: Read accumulated result from RF before writing
                cycle_rf_reads += 1
                # WS: PE to PE transfer for accumulation. For 1 OFMAP px, (r2c - 1) PE transfers of psums are required 
                cycle_pe_transfers += (window_size - 1)
            else: 
                # Code should not enter this part
                entry += "!, "
        
        # Update energy counters
        if energy_counter:
            energy_counter.increment_sram_write(cycle_sram_writes)
            energy_counter.increment_rf_read(cycle_rf_reads)
            energy_counter.increment_pe_data_transfer(cycle_pe_transfers)

        entry += "\n"
        outfile.write(entry)

    outfile.close()
    return cycle


# Trace generation for moving generated ofmap data in cases when only partial window fits
# This implementation prints out the ofmap pixel in the exact cycle it is generated
# Not used in scale sim at the moment. 
# SCALE sim waits till all the columns finish generating OFMAP.
def gen_trace_ofmap_partial_imm(
                        cycle = 0,
                        num_rows = 4, num_cols =4,
                        ofmap_base = 2000000,
                        num_ofmap_px = 16,
                        num_filter = 8,
                        filters_done = 0,
                        sram_write_trace_file = "sram_write.csv"
):
    outfile = open(sram_write_trace_file,'a')
    start_cycle = num_rows + cycle

    col_addr = []
    for col in range(int(num_cols)):
        a = (filters_done + col)
        col_addr.append(a)
    
    for tick in range(int(num_ofmap_px + num_cols)):
        cycle = start_cycle + tick

        entry = str(cycle) + ", "
        for col in range(int(num_cols)):
            # Condition to maintain skew
            if tick >= col and (tick - col)< num_ofmap_px:
                entry += str(col_addr[col]) + ", "
                col_addr[col] += num_filter
            else:
                entry += ", "
        
        entry += "\n"
        outfile.write(entry)

    outfile.close()


if __name__ == "__main__":
    h_h = 5 
    h_w = 5

    r_h = 2
    r_w = 2

    c = 2
    u =1

    m = 9

    dim_h = 16
    dim_v = 5

    sram_traffic(
        dimension_rows = dim_h,
        dimension_cols = dim_v,

        ifmap_h = h_h, ifmap_w = h_w,
        filt_h = r_h, filt_w = r_w, 
        num_channels = c,
        strides = u,

        num_filt = m
    )
