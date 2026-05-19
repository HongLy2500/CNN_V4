# collect_vivado_reports.tcl
# Run after implementation is complete/open in Vivado.
# Usage from Vivado Tcl console:
#   source eval/reports/collect_vivado_reports.tcl

set out_dir "eval/results/vgg16_conv13/vivado"
file mkdir $out_dir

report_utilization    -file "$out_dir/utilization.rpt"       -hierarchical
report_timing_summary -file "$out_dir/timing_summary.rpt"
report_power          -file "$out_dir/power.rpt"
report_clock_utilization -file "$out_dir/clock_utilization.rpt"

puts "Vivado reports written to $out_dir"
