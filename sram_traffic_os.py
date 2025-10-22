import math 
import csv
from tqdm import tqdm

neg_inf = -1 * math.pow(2,32)

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
    E_h = (ifmap_h - filt_h + strides) / strides
    E_w = (ifmap_w - filt_w + strides) / strides
    
    # Number of pixels in one convolution window
    px_per_conv_window = filt_h * filt_w * num_channels
    r2c = px_per_conv_window

    # Total number of ofmap px across all channels
    num_ofmap_px = E_h * E_w * num_filt
    e2  = E_h * E_w
    e2m = num_ofmap_px
    
    # Variables to calculate folds in runtime
    num_h_fold = math.ceil(e2/dimension_rows)
    num_v_fold = math.ceil(num_filt/dimension_cols)

    cycles = 0

    read_cycles, util, energy_counter = gen_read_trace(
                            cycle = cycles,
                            dim_rows = dimension_rows,
                            dim_cols = dimension_cols,
                            num_v_fold = int(num_v_fold),
                            num_h_fold = int(num_h_fold),
                            ifmap_h = ifmap_h, ifmap_w= ifmap_w,
                            filt_h= filt_h, filt_w= filt_w,
                            num_channels= num_channels, stride=strides,
                            ofmap_h= int(E_h), ofmap_w= int(E_w), num_filters = num_filt,
                            filt_base= filt_base, ifmap_base= ifmap_base,
                            sram_read_trace_file= sram_read_trace_file,
                            energy_counter= energy_counter
                            )

    write_cycles, energy_counter = gen_write_trace(
                        cycle = cycles,
                        dim_rows = dimension_rows,
                        dim_cols = dimension_cols,
                        #num_v_fold = int(num_v_fold),
                        #num_h_fold = int(num_h_fold),
                        ofmap_h = int(E_h), ofmap_w = int(E_w),
                        num_filters = num_filt,
                        ofmap_base = ofmap_base,
                        conv_window_size = r2c,
                        sram_write_trace_file = sram_write_trace_file,
                        energy_counter = energy_counter
                        )

    cycles = max(read_cycles, write_cycles)
    str_cycles = str(cycles)
    
    return(str_cycles, util, energy_counter)
# End of sram_traffic()

        
def gen_read_trace(
        cycle = 0,
        dim_rows = 4, 
        dim_cols = 4,
        num_v_fold = 1,
        num_h_fold = 1,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w =3,
        num_channels = 3, stride = 1,
        ofmap_h =5, ofmap_w = 5, num_filters = 8, 
        filt_base = 1000000, ifmap_base = 0,
        sram_read_trace_file = "sram_read.csv",
        energy_counter = None
        #sram_write_trace_file = "sram_write.csv"
):
    # Layer specific variables
    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels
    e2 = ofmap_h * ofmap_w
    #num_ofmap_px = e2 * num_filters
    
    # Tracking variables
    local_cycle     = 0
    #remaining_px    = e2           # Need tracking for individual v folds
    #remaining_px     = []
    remaining_filt  = num_filters
    ifmap_done      = False
    filt_done       = False
    row_base_addr   = []
    row_clk_offset  = []
    row_ofmap_idx   = []
    v_fold_row      = []
    col_base_addr   = []
    col_clk_offset  = []
    v_fold_col      = []
    h_fold_col      = []
    lane_done       = []
    v_fold_barrier  = []

    # Variables for utilization calculation
    rows_used = 0
    cols_used = 0
    util      = 0

    # This initialization assumes num_rows << num_ofmap_px
    # The assignment logic needs to be modified if that is not the case
    for r in range(dim_rows):
        base_row_id = math.floor(r / ofmap_w) * stride
        base_col_id = r % ofmap_w * stride
        base_addr  = base_row_id * hc + base_col_id * num_channels 

        if r < e2:
            clk_offset = r * -1             # Clock offset takes care of the skew due to store and forward
        else:
            clk_offset = neg_inf            # In case num_ofamp_px < dim_rows

        row_base_addr.append(base_addr)
        row_clk_offset.append(clk_offset)
        row_ofmap_idx.append(r)
        v_fold_row.append(0)
        v_fold_barrier.append(False)

    for c in range(dim_cols):
        base_addr = c * r2c

        # Anand: TODO
        if c < remaining_filt:
            clk_offset = c * -1
            lane_done.append(False)
        else:
            clk_offset = neg_inf
            lane_done.append(True)

        col_base_addr.append(base_addr)
        col_clk_offset.append(clk_offset)
        v_fold_col.append(0)
        h_fold_col.append(0)


    # Open tracefile for writing
    outfile     = open(sram_read_trace_file, 'w')
    #ofmap_out   = open(sram_write_trace_file, 'w')

    # Adding progress bar
    tot  = e2 * num_v_fold
    #print("Total = " + str(tot))
    pbar = tqdm(total=tot)

    # Generate traces here
    # The condition checks
    #       1)  if the all the input traces for last v fold is generated
    # and   2)  if all the filter traces have been generated
    #while(remaining_px[num_v_fold-1] > 0) or (filt_done == False):
    while(ifmap_done == False) or (filt_done == False):
        ifmap_read = ""
        filt_read  = ""
        rows_used = 0
        cols_used = 0
        
        # Count for this cycle
        cycle_sram_reads = 0
        cycle_rf_reads = 0
        cycle_rf_writes = 0
        cycle_alu_ops = 0
        cycle_pe_transfers = 0
        
        # Generate address for ifmap
        for r in range(dim_rows):

            if(row_clk_offset[r] >= 0):     # Take care of the skew

                inc = row_clk_offset[r]

                addr_row_offset = math.floor(inc / rc) * ifmap_w * num_channels
                addr_col_offset = inc % rc
                ifmap_addr = row_base_addr[r] + addr_row_offset + addr_col_offset 
                ifmap_read += str(int(ifmap_addr)) + ", "
                rows_used += 1
                
                # Count SRAM read for ifmap
                cycle_sram_reads += 1
                
            else:
                ifmap_read += ", "

            row_clk_offset[r] += 1

            if (row_clk_offset[r] > 0) and (row_clk_offset[r] % r2c == 0):   #Completed MAC for one OFMAP px
                
                row_ofmap_idx[r] += dim_rows
                ofmap_idx = row_ofmap_idx[r]

                # Update progress bar
                pbar.update(1)

                if ofmap_idx < e2:
                    row_clk_offset[r] = 0

                    base_row_id = math.floor(ofmap_idx / ofmap_w) * stride
                    base_col_id = ofmap_idx % ofmap_w * stride
                    base_addr  = base_row_id * hc + base_col_id * num_channels
                    row_base_addr[r] = base_addr

                else:
                    v_fold_row[r] += 1
                    #pbar.update(e2)

                    if(v_fold_row[r] < num_v_fold):
                        row_ofmap_idx[r]  = r

                        base_row_id = math.floor(r / ofmap_w) * stride
                        base_col_id = r % ofmap_w * stride
                        base_addr  = base_row_id * hc + base_col_id * num_channels
                        row_base_addr[r]  = base_addr

                        # Stall this col from proceeding until all the rows reach the v_fold boundary
                        if (r != 0) and ((v_fold_row[r] > v_fold_row[r-1]) or (v_fold_barrier[r-1] == True)):
                            row_clk_offset[r] = neg_inf
                            v_fold_barrier[r] = True
                        else:
                            row_clk_offset[r] = 0

                    else:
                        row_clk_offset[r] = neg_inf

        # Get out of the barrier one by one
        # IMPORTANT: The barrier insertion and recovery is in separate loops to ensure that
        #            in a given clock cycle insertion for all rows strictly happen before the release.
        #            The flag ensures only one col is released per cycle
        # Since indx 0 never enters the barrier, this should work fine
        flag = False
        for r in range(dim_rows):
            if v_fold_barrier[r] and flag==False:
                if (v_fold_row[r] == v_fold_row[r-1]) and (v_fold_barrier[r-1] == False):
                    v_fold_barrier[r] = False
                    flag = True
                    row_clk_offset[r] = row_clk_offset[r-1] -1

        # Check if all input traces are done
        ifmap_done = True
        for r in range(dim_rows):
            if row_clk_offset[r] > 0:
                ifmap_done = False

        # Generate address for filters
        for c in range(dim_cols):
            if(col_clk_offset[c] >= 0):     # Take care of the skew
                inc = col_clk_offset[c]
                
                filt_addr = col_base_addr[c] + inc + filt_base 
                filt_read += str(filt_addr) + ", "
                cols_used += 1
                
                # Count SRAM read for filter
                cycle_sram_reads += 1
                
            else:
                filt_read += ", "

            col_clk_offset[c] += 1

            if(col_clk_offset[c] > 0) and (col_clk_offset[c] % r2c == 0):

                # Get the v fold this col is working on and check the status of input trace generation
                #rem_px = remaining_px[v_fold_col[c]]

                #Tracking on the basis of h_folds
                h_fold_col[c] += 1

                # Anand: Check if all the input traces are generated for the given v fold
                if (h_fold_col[c] < num_h_fold):
                    col_clk_offset[c] = 0
                else:
                    v_fold_col[c] += 1
                    filt_id = v_fold_col[c] * dim_cols + c

                    # All filters might not be active in the last fold
                    # Adding the filter ID check to ensure only valid cols are active
                    if(v_fold_col[c] < num_v_fold) and (filt_id < num_filters):
                        col_clk_offset[c] = 0
                        h_fold_col[c] = 0

                        base = filt_id * r2c
                        col_base_addr[c] = base

                    else:
                        col_clk_offset[c] = neg_inf
                        lane_done[c] = True

        # Check if all filter traces are generated
        filt_done = True
        for c in range(dim_cols):
            if lane_done[c] == False:
                filt_done = False

        # Count MAC operations and RF reads for this cycle
        active_pes = rows_used * cols_used
        if active_pes > 0:
            # Each active PE performs MAC operations
            cycle_alu_ops = active_pes  # One MAC per active PE
            # Each PE reads from RF for MAC operation
            cycle_rf_reads = active_pes * 2  # Read ifmap and filter data
            # PE data transfer (for output stationary, intermediate results flow between PEs)
            cycle_pe_transfers = active_pes // 2  # Approximate PE-to-PE transfers
        
        # Update energy counters
        if energy_counter:
            energy_counter.increment_sram_read(cycle_sram_reads)
            energy_counter.increment_rf_read(cycle_rf_reads)
            energy_counter.increment_rf_write(cycle_rf_writes)
            energy_counter.increment_alu_operations(cycle_alu_ops)
            energy_counter.increment_pe_data_transfer(cycle_pe_transfers)
                                                
        # Write to trace file
        global_cycle = cycle + local_cycle
        entry = str(global_cycle) + ", " + ifmap_read + filt_read + "\n"
        outfile.write(entry)

        this_util = (rows_used * cols_used) / (dim_rows * dim_cols)
        util += this_util

        # Update tracking variables
        local_cycle += 1

    pbar.close()
    outfile.close()
    #ofmap_out.close()

    util_perc = (util / local_cycle) * 100

    return (local_cycle + cycle), util_perc, energy_counter
# End of gen_read_trace()


def gen_write_trace(
        cycle = 0,
        dim_rows = 4,
        dim_cols = 4,
        #num_v_fold = 1,
        #num_h_fold = 1,
        ofmap_h = 5, ofmap_w = 5,
        num_filters = 4,
        ofmap_base = 2000000,
        conv_window_size = 9,                      # The number of pixels in a convolution window
        sram_write_trace_file = "sram_write.csv",
        energy_counter = None
):

    # Layer specific variables
    r2c = conv_window_size
    e2  = ofmap_h * ofmap_w

    # Tracking variables
    id_row = []             # List of OFMAP ID for each row
    id_col = []             # List of filter ID for each col
    base_addr_col =[]       # Starting address of each output channel
    remaining_px  = e2
    remaining_filt= num_filters
    active_row = min(dim_rows, e2)
    active_col = min(dim_cols, num_filters)
    local_cycle = 0
    sticky_flag = False     # This flag is in place to fix the OFMAP cycle shaving bug

    for r in range(active_row):
        id_row.append(r)

    for c in range(active_col):
        id_col.append(c)

        base_col = c
        base_addr_col.append(base_col)

    #Open the file for writing
    outfile = open(sram_write_trace_file,"w")

    #This is the cycle when all the OFMAP elements in the first col become available
    local_cycle = r2c + active_col - 1

    while (remaining_px > 0) or (remaining_filt > 0):

        active_row = min(dim_rows, remaining_px)

        for r in range(active_row):
            local_px = id_row[r]
            remaining_px -= 1
            id_row[r] += active_row     # Taking care of horizontal fold

            ofmap_trace = ""
            sram_write_count = 0
            rf_write_count = 0
            
            for c in range(active_col):
                addr = ofmap_base + base_addr_col[c] + local_px * num_filters
                ofmap_trace += str(addr) + ", "
                
                # Count SRAM write operations
                sram_write_count += 1
                # Count RF read (reading accumulated result from RF before writing to SRAM)
                rf_write_count += 1
            
            # Update energy counters
            if energy_counter:
                energy_counter.increment_sram_write(sram_write_count)
                energy_counter.increment_rf_write(rf_write_count)

            # Write the generated traces to the file
            entry = str(local_cycle + r) + ", " + ofmap_trace + "\n"
            outfile.write(entry)

        # Take care of the vertical fold
        if remaining_px == 0:
            remaining_filt -= active_col

            # In case of vertical fold we have to track when the output of (0,0) is generated
            # Shifting back local cycles to capture the last OFMAP generation in (0,0) for this fold
            last_fold_cycle   = local_cycle + active_row
            local_cycle -= (active_row + active_col - 1)
            sticky_flag = True

            # There are more OFMAP channels to go
            if remaining_filt > 0:
                remaining_px = e2
                last_active_col = active_col
                active_col = min(remaining_filt, dim_cols)

                # Reassign col base addresses
                for c in range(active_col):
                    base_addr_col[c] += last_active_col

                active_row = min(dim_rows, remaining_px)
                # Reassign row base addresses
                for r in range(active_row):
                    id_row[r] = r

                local_cycle += r2c + active_col
                if local_cycle < last_fold_cycle:
                    local_cycle = last_fold_cycle


            else:   # Restore the local cycle to return to the main function
                local_cycle = last_fold_cycle
                #local_cycle += (active_row + active_col)
                #sticky_flag = False

        else:   # If this is not a vertical fold then it is business as usual
            local_cycle += max(r2c, active_row)

    outfile.close()

    #if sticky_flag:
    #    local_cycle += (active_row + active_col)
    #    sticky_flag = False

    return (local_cycle + cycle), energy_counter
# End of gen_write_trace()


if __name__ == "__main__":
   print("Testing Energy Counting for Output Stationary Dataflow")
   print("="*60)
   
   result = sram_traffic(
       dimension_rows = 8,
       dimension_cols = 4,
       ifmap_h = 7, ifmap_w = 7,
       filt_h = 2, filt_w = 2,
       num_channels = 1, strides = 1,
       num_filt = 7
   )
   
   if len(result) >= 3:
       cycles, util, energy_counter = result[0], result[1], result[2]
       print(f"Total cycles: {cycles}")
       print(f"Utilization: {util}%")
       if energy_counter:
           print(f"Total Energy: {energy_counter.calculate_total_energy():.1f} pJ")
           energy_counter.save_to_csv("test_energy_breakdown.csv")
           print("Energy breakdown saved to test_energy_breakdown.csv")
   
   print("="*60)
