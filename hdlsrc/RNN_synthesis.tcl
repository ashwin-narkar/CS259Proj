################################################################################
# DESIGN COMPILER:  Logic Synthesis Tool                                       #
################################################################################

remove_design -all
set hdlin_vrlg_std "2001"

#Library Paths
set search_path "$search_path . ./verilog /w/apps2/public.2/tech/synopsys/32-28nm/SAED32_EDK/lib/stdcell_rvt/db_nldm" 
set target_library "saed32rvt_ff1p16vn40c.db saed32rvt_ss0p95v125c.db"
set link_library "* saed32rvt_ff1p16vn40c.db saed32rvt_ss0p95v125c.db dw_foundation.sldb"
set synthetic_library "dw_foundation.sldb"

#Work Path
define_design_lib WORK -path ./WORK
set alib_library_analysis_path �./alib-52/�

#Read Verilog Files
analyze -format verilog {nfp_add_single.v nfp_mul_single.v nfp_tanh_single.v RAM.v RNN.v RNN_Node.v SynLib.v}
#analyze -format verilog {generated_image_classifier.v}
set DESIGN_NAME RNN

elaborate $DESIGN_NAME
current_design $DESIGN_NAME
link

#Setup Operating Conditions
set_operating_conditions -min ff1p16vn40c -max ss0p95v125c

#Clock and Delay Setup (Substituted with our clock name)
set Tclk 2.0
set TCU 0.1
set IN_DEL 0.6
set IN_DEL_MIN 0.3
set OUT_DEL 0.6
set OUT_DEL_MIN 0.3
set ALL_IN_BUT_CLK [remove_from_collection [all_inputs] "clk"]

create_clock -name "clk" -period $Tclk [get_ports "clk"]
set_fix_hold clk
set_dont_touch_network [get_clocks "clk"]
set_clock_uncertainty $TCU [get_clocks "clk"]

set_input_delay $IN_DEL -clock "clk" $ALL_IN_BUT_CLK
set_input_delay -min $IN_DEL_MIN -clock "clk" $ALL_IN_BUT_CLK
set_output_delay $OUT_DEL -clock "clk" [all_outputs]
set_output_delay -min $OUT_DEL_MIN -clock "clk" [all_outputs]

set_max_area 0.0


ungroup -flatten -all
uniquify

compile -only_design_rule
compile -map high
compile -boundary_optimization
compile -only_hold_time

report_timing -path full -delay min -max_paths 10 -nworst 2 > Design.holdtiming
report_timing -path full -delay max -max_paths 10 -nworst 2 > Design.setuptiming
report_area -hierarchy > Design.area
report_power -hier -hier_level 2 > Design.power
report_resources > Design.resources
report_constraint -verbose > Design.constraint
check_design > Design.check_design
check_timing > Design.check_timing

write -hierarchy -format verilog -output $DESIGN_NAME.vg
write_sdf -version 1.0 -context verilog $DESIGN_NAME.sdf
set_propagated_clock [all_clocks]
write_sdc $DESIGN_NAME.sdc
write_file -output $DESIGN_NAME.ddc