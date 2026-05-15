# run_kv260_eval.tcl
# -----------------------------------------------------------------------------
# KV260 CNN_V4 evaluation runner for XSCT.
#
# Purpose:
#   1) Write IFM and weight hex files into the reserved PS DDR window.
#   2) Optionally start/poll the accelerator through a simple memory-mapped
#      control register map if your Vivado block design provides one.
#   3) Read OFM words back from DDR into a hex file for compare_ofm.py.
#
# IMPORTANT:
#   The current kv260_cnn_smoke_top.sv in CNN_V4 exposes run/status as top-level
#   VIO/GPIO-style signals, not as a fixed AXI-Lite software register map.
#   Therefore this script supports:
#
#     -ctrl-mode manual:
#         Script writes IFM/WGT and clears OFM, then waits for you to start the
#         accelerator manually through VIO/GPIO. Press ENTER after done is high.
#
#     -ctrl-mode reg:
#         Script assumes YOU added/connected a memory-mapped control/status
#         register block. Default map is documented below and can be overridden.
#
# DDR word map used by the current RTL headers:
#   DDR word address space, not byte address space:
#     IFM word base  = 0x00000
#     WGT word base  = 0x08000
#     OFM word base  = 0x10000
#     RSVD word base = 0x18000
#
# Default byte mapping for the 32-bit smoke/KV260 wrapper:
#   AXI_DDR_BASE_ADDR = 0x0000_7000_0000
#   WORD_BYTES        = 4
#   IFM byte addr     = AXI_DDR_BASE + 0x00000 * 4 = 0x70000000
#   WGT byte addr     = AXI_DDR_BASE + 0x08000 * 4 = 0x70020000
#   OFM byte addr     = AXI_DDR_BASE + 0x10000 * 4 = 0x70040000
#
# Example, manual VIO/GPIO start:
#   xsct eval/board/run_kv260_eval.tcl \
#     -ifm eval/assets/vgg16_conv13/inputs/img0000/input_uint8_hwc_u32le.hex \
#     -wgt eval/assets/vgg16_conv13/weights/all_weights_linear_u32le.hex \
#     -out eval/results/vgg16_conv13/img0000/fpga_ofm_readback.hex \
#     -ofm-words 1024 \
#     -ctrl-mode manual
#
# Example, memory-mapped control:
#   xsct eval/board/run_kv260_eval.tcl \
#     -ifm .../ifm.hex \
#     -wgt .../weights.hex \
#     -out .../fpga_ofm_readback.hex \
#     -ofm-words 1024 \
#     -ctrl-mode reg \
#     -ctrl-base 0xA0000000
#
# Default control register map for -ctrl-mode reg:
#   ctrl_base + 0x00 : CONTROL, writable
#       bit 0 = run
#       bit 1 = soft_reset_n
#       bit 2 = abort
#
#   ctrl_base + 0x04 : STATUS, readable
#       bit 0 = done
#       bit 1 = busy
#       bit 2 = error
#
# If your AXI-GPIO/control block uses a different map, override the offsets/masks.
# -----------------------------------------------------------------------------

# --------------------------
# Utility functions
# --------------------------
proc usage {} {
    puts ""
    puts "Usage:"
    puts "  xsct run_kv260_eval.tcl \\"
    puts "    -ifm <ifm_hex> \\"
    puts "    -wgt <weight_hex> \\"
    puts "    -out <fpga_ofm_readback_hex> \\"
    puts "    -ofm-words <num_words> \\"
    puts "    ?options?"
    puts ""
    puts "Required:"
    puts "  -ifm <path>              IFM input hex, one 32-bit word per line"
    puts "  -wgt <path>              Weight hex, one 32-bit word per line"
    puts "  -out <path>              Output OFM readback hex path"
    puts "  -ofm-words <N>           Number of 32-bit OFM words to read back"
    puts ""
    puts "Useful options:"
    puts "  -base-addr <addr>        AXI DDR base byte address. Default 0x70000000"
    puts "  -word-bytes <N>          Bytes per DDR word. Default 4"
    puts "  -ifm-base-word <addr>    IFM word base. Default 0x00000"
    puts "  -wgt-base-word <addr>    WGT word base. Default 0x08000"
    puts "  -ofm-base-word <addr>    OFM word base. Default 0x10000"
    puts "  -clear-ofm-words <N>     OFM words to clear before run. Default = -ofm-words"
    puts "  -skip-ifm-write          Do not write IFM"
    puts "  -skip-wgt-write          Do not write weights"
    puts "  -skip-ofm-clear          Do not clear OFM"
    puts "  -skip-run                Do not start/poll; only write/read DDR"
    puts "  -target-index <N>        Optional XSCT target index"
    puts ""
    puts "Control options:"
    puts "  -ctrl-mode manual|reg    Default manual"
    puts "  -ctrl-base <addr>        Required for -ctrl-mode reg unless default is OK"
    puts "  -ctrl-offset <off>       CONTROL offset. Default 0x00"
    puts "  -status-offset <off>     STATUS offset. Default 0x04"
    puts "  -run-mask <mask>         CONTROL run bit. Default 0x1"
    puts "  -soft-reset-mask <mask>  CONTROL soft_reset_n bit. Default 0x2"
    puts "  -abort-mask <mask>       CONTROL abort bit. Default 0x4"
    puts "  -done-mask <mask>        STATUS done bit. Default 0x1"
    puts "  -busy-mask <mask>        STATUS busy bit. Default 0x2"
    puts "  -error-mask <mask>       STATUS error bit. Default 0x4"
    puts "  -poll-timeout-ms <ms>    Default 60000"
    puts "  -poll-interval-ms <ms>   Default 20"
    puts ""
    exit 1
}

proc arg_value {args_ref name default} {
    upvar $args_ref args
    set idx [lsearch -exact $args $name]
    if {$idx < 0} {
        return $default
    }
    if {$idx == [expr {[llength $args] - 1}]} {
        error "Missing value after $name"
    }
    return [lindex $args [expr {$idx + 1}]]
}

proc has_flag {args_ref name} {
    upvar $args_ref args
    expr {[lsearch -exact $args $name] >= 0}
}

proc parse_int {s} {
    # Tcl expr handles decimal and 0x... strings.
    return [expr {$s + 0}]
}

proc ensure_parent_dir {path} {
    set dir [file dirname $path]
    if {$dir ne "." && ![file exists $dir]} {
        file mkdir $dir
    }
}

proc timestamp {} {
    return [clock format [clock seconds] -format "%Y-%m-%d %H:%M:%S"]
}

proc log {msg} {
    puts "[timestamp] $msg"
}

proc clean_hex_line_to_word {line} {
    # Accept clean lines such as:
    #   000000ff
    #   0x000000ff
    # Also tolerate comments after # or //.
    regsub {#.*$} $line "" line
    regsub {//.*$} $line "" line
    set line [string trim $line]
    if {$line eq ""} {
        return ""
    }

    # If the line contains a "data=0x..." field, prefer that.
    if {[regexp -nocase {data\s*=\s*0x([0-9a-f]+)} $line -> hx]} {
        return [expr {0x$hx & 0xFFFFFFFF}]
    }

    # Otherwise find first hex token.
    if {[regexp -nocase {0x([0-9a-f]+)} $line -> hx]} {
        return [expr {0x$hx & 0xFFFFFFFF}]
    }

    if {[regexp -nocase {^[0-9a-f]+$} $line]} {
        return [expr {0x$line & 0xFFFFFFFF}]
    }

    error "Cannot parse hex word from line: $line"
}

proc read_hex_words {path} {
    if {![file exists $path]} {
        error "Hex file not found: $path"
    }

    set fh [open $path r]
    set words {}
    set line_no 0

    while {[gets $fh line] >= 0} {
        incr line_no
        set word [clean_hex_line_to_word $line]
        if {$word eq ""} {
            continue
        }
        lappend words $word
    }

    close $fh
    return $words
}

proc write_ddr_words_from_hex {path byte_addr max_words} {
    set words [read_hex_words $path]
    set n [llength $words]

    if {$max_words >= 0 && $n > $max_words} {
        error "File $path has $n words, exceeds region limit $max_words words"
    }

    log "Writing $n words from $path to DDR byte address [format 0x%08X $byte_addr]"

    set addr $byte_addr
    set i 0

    foreach word $words {
        mwr [format 0x%08X $addr] [format 0x%08X $word]
        incr addr 4
        incr i

        if {($i % 4096) == 0} {
            log "  wrote $i / $n words"
        }
    }

    log "Done writing $n words"
    return $n
}

proc clear_ddr_words {byte_addr num_words} {
    if {$num_words <= 0} {
        return
    }

    log "Clearing $num_words words at DDR byte address [format 0x%08X $byte_addr]"

    set addr $byte_addr
    for {set i 0} {$i < $num_words} {incr i} {
        mwr [format 0x%08X $addr] 0x00000000
        incr addr 4

        if {(($i + 1) % 4096) == 0} {
            log "  cleared [expr {$i + 1}] / $num_words words"
        }
    }

    log "Done clearing OFM region"
}

proc mrd_word {byte_addr} {
    # XSCT supports mrd -value on modern versions. Fall back to parsing if needed.
    set addr_str [format 0x%08X $byte_addr]
    if {[catch {set v [mrd -value $addr_str]} err]} {
        set raw [mrd $addr_str]
        if {[regexp -nocase {:\s*([0-9a-f]{8})} $raw -> hx]} {
            return [expr {0x$hx & 0xFFFFFFFF}]
        }
        if {[regexp -nocase {0x[0-9a-f]+\s+([0-9a-f]{8})} $raw -> hx]} {
            return [expr {0x$hx & 0xFFFFFFFF}]
        }
        error "Could not read/parse word at $addr_str. mrd error/raw: $err / $raw"
    } else {
        return [expr {$v & 0xFFFFFFFF}]
    }
}

proc read_ddr_words_to_hex {byte_addr num_words out_path} {
    ensure_parent_dir $out_path

    log "Reading $num_words words from DDR byte address [format 0x%08X $byte_addr] to $out_path"

    set fh [open $out_path w]
    set addr $byte_addr

    for {set i 0} {$i < $num_words} {incr i} {
        set v [mrd_word $addr]
        puts $fh [format "%08X" $v]
        incr addr 4

        if {(($i + 1) % 4096) == 0} {
            log "  read [expr {$i + 1}] / $num_words words"
        }
    }

    close $fh
    log "Done reading OFM"
}

proc ctrl_reg_run_and_poll {ctrl_base ctrl_off status_off run_mask soft_reset_mask abort_mask done_mask busy_mask error_mask timeout_ms interval_ms} {
    set ctrl_addr   [expr {$ctrl_base + $ctrl_off}]
    set status_addr [expr {$ctrl_base + $status_off}]

    log "Using memory-mapped control"
    log "  CONTROL addr = [format 0x%08X $ctrl_addr]"
    log "  STATUS  addr = [format 0x%08X $status_addr]"

    # Release soft reset, abort low, run low.
    set ctrl_idle [expr {$soft_reset_mask}]
    mwr [format 0x%08X $ctrl_addr] [format 0x%08X $ctrl_idle]
    after 10

    # Pulse/level run high. The kv260 wrapper accepts a run level and starts once.
    set ctrl_run [expr {$soft_reset_mask | $run_mask}]
    log "Asserting run"
    mwr [format 0x%08X $ctrl_addr] [format 0x%08X $ctrl_run]

    set elapsed 0
    set final_status 0

    while {$elapsed <= $timeout_ms} {
        set st [mrd_word $status_addr]
        set final_status $st

        if {[expr {$st & $error_mask}] != 0} {
            log "ERROR bit set in status: [format 0x%08X $st]"
            # Lower run before failing.
            mwr [format 0x%08X $ctrl_addr] [format 0x%08X $ctrl_idle]
            error "Accelerator reported error"
        }

        if {[expr {$st & $done_mask}] != 0} {
            log "DONE observed. status=[format 0x%08X $st]"
            mwr [format 0x%08X $ctrl_addr] [format 0x%08X $ctrl_idle]
            return $st
        }

        after $interval_ms
        set elapsed [expr {$elapsed + $interval_ms}]
    }

    # Lower run after timeout.
    mwr [format 0x%08X $ctrl_addr] [format 0x%08X $ctrl_idle]
    error "Timeout waiting for done after ${timeout_ms} ms. Last status=[format 0x%08X $final_status]"
}

proc manual_run_wait {} {
    puts ""
    puts "----------------------------------------------------------------------"
    puts "Manual control mode:"
    puts "  1) IFM/WGT have been written and OFM has been cleared."
    puts "  2) Start the accelerator manually through VIO/GPIO:"
    puts "       soft_reset_n = 1"
    puts "       abort        = 0"
    puts "       run          = 1"
    puts "  3) Wait until done=1 and error=0."
    puts "  4) Press ENTER here to read back OFM."
    puts "----------------------------------------------------------------------"
    puts -nonewline "Press ENTER after the accelerator is DONE: "
    flush stdout
    gets stdin
}

# --------------------------
# Main
# --------------------------
set args $argv

if {[has_flag args "-h"] || [has_flag args "--help"]} {
    usage
}

set ifm_hex        [arg_value args "-ifm" ""]
set wgt_hex        [arg_value args "-wgt" ""]
set out_hex        [arg_value args "-out" ""]
set ofm_words_s    [arg_value args "-ofm-words" ""]

if {$ifm_hex eq "" || $wgt_hex eq "" || $out_hex eq "" || $ofm_words_s eq ""} {
    usage
}

set base_addr      [parse_int [arg_value args "-base-addr" "0x70000000"]]
set word_bytes     [parse_int [arg_value args "-word-bytes" "4"]]
set ifm_base_word  [parse_int [arg_value args "-ifm-base-word" "0x00000"]]
set wgt_base_word  [parse_int [arg_value args "-wgt-base-word" "0x08000"]]
set ofm_base_word  [parse_int [arg_value args "-ofm-base-word" "0x10000"]]
set ifm_size_words [parse_int [arg_value args "-ifm-size-words" "0x08000"]]
set wgt_size_words [parse_int [arg_value args "-wgt-size-words" "0x08000"]]
set ofm_size_words [parse_int [arg_value args "-ofm-size-words" "0x08000"]]
set ofm_words      [parse_int $ofm_words_s]
set clear_words_s  [arg_value args "-clear-ofm-words" ""]
if {$clear_words_s eq ""} {
    set clear_ofm_words $ofm_words
} else {
    set clear_ofm_words [parse_int $clear_words_s]
}

set ctrl_mode      [string tolower [arg_value args "-ctrl-mode" "manual"]]
set ctrl_base      [parse_int [arg_value args "-ctrl-base" "0x00000000"]]
set ctrl_off       [parse_int [arg_value args "-ctrl-offset" "0x00"]]
set status_off     [parse_int [arg_value args "-status-offset" "0x04"]]
set run_mask       [parse_int [arg_value args "-run-mask" "0x1"]]
set soft_rst_mask  [parse_int [arg_value args "-soft-reset-mask" "0x2"]]
set abort_mask     [parse_int [arg_value args "-abort-mask" "0x4"]]
set done_mask      [parse_int [arg_value args "-done-mask" "0x1"]]
set busy_mask      [parse_int [arg_value args "-busy-mask" "0x2"]]
set error_mask     [parse_int [arg_value args "-error-mask" "0x4"]]
set timeout_ms     [parse_int [arg_value args "-poll-timeout-ms" "60000"]]
set interval_ms    [parse_int [arg_value args "-poll-interval-ms" "20"]]
set target_index_s [arg_value args "-target-index" ""]

set skip_ifm       [has_flag args "-skip-ifm-write"]
set skip_wgt       [has_flag args "-skip-wgt-write"]
set skip_clear     [has_flag args "-skip-ofm-clear"]
set skip_run       [has_flag args "-skip-run"]

if {$word_bytes != 4} {
    error "This script currently writes/reads 32-bit words only. word-bytes must be 4."
}

if {$ofm_words <= 0} {
    error "-ofm-words must be > 0"
}
if {$ofm_words > $ofm_size_words} {
    error "-ofm-words $ofm_words exceeds OFM region size $ofm_size_words"
}

set ifm_byte_addr [expr {$base_addr + $ifm_base_word * $word_bytes}]
set wgt_byte_addr [expr {$base_addr + $wgt_base_word * $word_bytes}]
set ofm_byte_addr [expr {$base_addr + $ofm_base_word * $word_bytes}]

# Optional target selection.
if {$target_index_s ne ""} {
    set target_index [parse_int $target_index_s]
    log "Selecting XSCT target index $target_index"
    targets $target_index
}

log "KV260 CNN_V4 evaluation script"
log "Base DDR byte address : [format 0x%08X $base_addr]"
log "IFM byte address      : [format 0x%08X $ifm_byte_addr]"
log "WGT byte address      : [format 0x%08X $wgt_byte_addr]"
log "OFM byte address      : [format 0x%08X $ofm_byte_addr]"
log "OFM readback words    : $ofm_words"
log "Control mode          : $ctrl_mode"

if {!$skip_ifm} {
    write_ddr_words_from_hex $ifm_hex $ifm_byte_addr $ifm_size_words
} else {
    log "Skipping IFM write"
}

if {!$skip_wgt} {
    write_ddr_words_from_hex $wgt_hex $wgt_byte_addr $wgt_size_words
} else {
    log "Skipping weight write"
}

if {!$skip_clear} {
    clear_ddr_words $ofm_byte_addr $clear_ofm_words
} else {
    log "Skipping OFM clear"
}

if {!$skip_run} {
    if {$ctrl_mode eq "manual"} {
        manual_run_wait
    } elseif {$ctrl_mode eq "reg"} {
        ctrl_reg_run_and_poll \
            $ctrl_base $ctrl_off $status_off \
            $run_mask $soft_rst_mask $abort_mask \
            $done_mask $busy_mask $error_mask \
            $timeout_ms $interval_ms
    } else {
        error "Unsupported -ctrl-mode '$ctrl_mode'. Use manual or reg."
    }
} else {
    log "Skipping run/poll"
}

read_ddr_words_to_hex $ofm_byte_addr $ofm_words $out_hex

log "All done."
log "OFM readback written to: $out_hex"
