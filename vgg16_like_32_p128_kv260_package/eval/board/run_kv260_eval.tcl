# run_kv260_eval.tcl for CNN_V4 VGG16-like 32 P128/PV8 wide-word evaluation
# -----------------------------------------------------------------------------
# Writes/reads generated DDR hex files whose line width equals -word-bytes.
# For this profile, one logical DDR word is 128 bits = 16 bytes, but XSCT mwr/mrd
# are performed as 32-bit chunks at byte_addr + 0, +4, +8, +12.
# -----------------------------------------------------------------------------

proc usage {} {
    puts "Usage: xsct run_kv260_eval.tcl -ifm <hex> -wgt <hex> -out <hex> -ofm-words <N> ?options?"
    puts "Defaults for VGG16-LIKE-32-P128: -word-bytes 16 -base-addr 0x70000000 -ifm-base-word 0x000000 -wgt-base-word 0x010000 -ofm-base-word 0x020000"
    puts "Control: -ctrl-mode manual|reg. manual writes DDR, clears OFM, waits for you to run via VIO/GPIO, then reads OFM."
    exit 1
}
proc arg_value {args_ref name default} { upvar $args_ref args; set idx [lsearch -exact $args $name]; if {$idx < 0} {return $default}; if {$idx == [expr {[llength $args]-1}]} {error "Missing value after $name"}; return [lindex $args [expr {$idx+1}]] }
proc has_flag {args_ref name} { upvar $args_ref args; expr {[lsearch -exact $args $name] >= 0} }
proc parse_int {s} { expr {$s + 0} }
proc timestamp {} { clock format [clock seconds] -format "%Y-%m-%d %H:%M:%S" }
proc log {msg} { puts "[timestamp] $msg" }
proc ensure_parent_dir {path} { set dir [file dirname $path]; if {$dir ne "." && ![file exists $dir]} { file mkdir $dir } }

proc clean_hex_to_str {line} {
    regsub {#.*$} $line "" line
    regsub {//.*$} $line "" line
    set line [string trim $line]
    if {$line eq ""} { return "" }
    if {[regexp -nocase {data\s*=\s*0x([0-9a-f]+)} $line -> hx]} { return [string toupper $hx] }
    if {[regexp -nocase {0x([0-9a-f]+)} $line -> hx]} { return [string toupper $hx] }
    if {[regexp -nocase {^[0-9a-f]+$} $line]} { return [string toupper $line] }
    error "Cannot parse hex word from line: $line"
}
proc read_hex_word_strings {path} {
    if {![file exists $path]} { error "Hex file not found: $path" }
    set fh [open $path r]
    set words {}
    while {[gets $fh line] >= 0} {
        set hx [clean_hex_to_str $line]
        if {$hx eq ""} { continue }
        lappend words $hx
    }
    close $fh
    return $words
}
proc hex_pad_to_bytes {hx word_bytes} {
    set hx [regsub -all {_} $hx ""]
    set max_digits [expr {$word_bytes * 2}]
    if {[string length $hx] > $max_digits} {
        error "Hex word $hx has more than $max_digits digits for word_bytes=$word_bytes"
    }
    while {[string length $hx] < $max_digits} { set hx "0$hx" }
    return $hx
}
proc word_hex_to_u32_chunks_le {hx word_bytes} {
    set hx [hex_pad_to_bytes $hx $word_bytes]
    set chunks {}
    # rightmost 8 hex digits are bits [31:0], written to lowest address.
    for {set pos [expr {[string length $hx]-8}]} {$pos >= 0} {set pos [expr {$pos-8}]} {
        set chunk [string range $hx $pos [expr {$pos+7}]]
        lappend chunks [expr {0x$chunk & 0xFFFFFFFF}]
    }
    return $chunks
}
proc u32_chunks_to_word_hex {chunks word_bytes} {
    set hx ""
    # Highest-address chunk is most significant. chunks are read low to high.
    for {set i [expr {[llength $chunks]-1}]} {$i >= 0} {incr i -1} {
        append hx [format "%08X" [lindex $chunks $i]]
    }
    set max_digits [expr {$word_bytes*2}]
    while {[string length $hx] < $max_digits} { set hx "0$hx" }
    return $hx
}
proc write_ddr_words_from_hex {path byte_addr word_bytes max_words} {
    set words [read_hex_word_strings $path]
    set n [llength $words]
    if {$max_words >= 0 && $n > $max_words} { error "$path has $n words, exceeds region limit $max_words" }
    if {[expr {$word_bytes % 4}] != 0} { error "word_bytes must be a multiple of 4" }
    log "Writing $n wide words from $path to byte address [format 0x%08X $byte_addr] word_bytes=$word_bytes"
    set i 0
    foreach hx $words {
        set chunks [word_hex_to_u32_chunks_le $hx $word_bytes]
        set addr [expr {$byte_addr + $i*$word_bytes}]
        for {set j 0} {$j < [llength $chunks]} {incr j} {
            set a [expr {$addr + $j*4}]
            mwr [format 0x%08X $a] [format 0x%08X [lindex $chunks $j]]
        }
        incr i
        if {($i % 4096) == 0} { log "  wrote $i / $n words" }
    }
    log "Done writing $n wide words"
}
proc clear_ddr_words {byte_addr word_bytes num_words} {
    if {$num_words <= 0} { return }
    log "Clearing $num_words wide words at byte address [format 0x%08X $byte_addr] word_bytes=$word_bytes"
    set chunks [expr {$word_bytes/4}]
    for {set i 0} {$i < $num_words} {incr i} {
        set addr [expr {$byte_addr + $i*$word_bytes}]
        for {set j 0} {$j < $chunks} {incr j} { mwr [format 0x%08X [expr {$addr+$j*4}]] 0x00000000 }
        if {($i % 4096) == 0 && $i > 0} { log "  cleared $i / $num_words words" }
    }
}
proc mrd_u32 {byte_addr} {
    set a [format 0x%08X $byte_addr]
    if {[catch {set v [mrd -value $a]} err]} {
        set raw [mrd $a]
        if {[regexp -nocase {0x[0-9a-f]+\s+0x([0-9a-f]+)} $raw -> hx]} { return [expr {0x$hx & 0xFFFFFFFF}] }
        error "Could not read $a: $err / $raw"
    }
    return [expr {$v & 0xFFFFFFFF}]
}
proc read_ddr_words_to_hex {byte_addr word_bytes num_words out_path} {
    ensure_parent_dir $out_path
    set fh [open $out_path w]
    set chunks_per [expr {$word_bytes/4}]
    log "Reading $num_words wide words from byte address [format 0x%08X $byte_addr] to $out_path"
    for {set i 0} {$i < $num_words} {incr i} {
        set addr [expr {$byte_addr + $i*$word_bytes}]
        set chunks {}
        for {set j 0} {$j < $chunks_per} {incr j} { lappend chunks [mrd_u32 [expr {$addr+$j*4}]] }
        puts $fh [u32_chunks_to_word_hex $chunks $word_bytes]
        if {($i % 4096) == 0 && $i > 0} { log "  read $i / $num_words words" }
    }
    close $fh
}
proc start_and_poll_reg {ctrl_base ctrl_off status_off run_mask soft_reset_mask abort_mask done_mask busy_mask error_mask timeout_ms poll_ms} {
    set ctrl_addr [expr {$ctrl_base + $ctrl_off}]
    set status_addr [expr {$ctrl_base + $status_off}]
    mwr [format 0x%08X $ctrl_addr] [format 0x%08X $soft_reset_mask]
    after 10
    mwr [format 0x%08X $ctrl_addr] [format 0x%08X [expr {$soft_reset_mask | $run_mask}]]
    set waited 0
    while {$waited < $timeout_ms} {
        set st [mrd_u32 $status_addr]
        if {($st & $error_mask) != 0} { error "Accelerator error status=[format 0x%08X $st]" }
        if {($st & $done_mask) != 0} { mwr [format 0x%08X $ctrl_addr] [format 0x%08X $soft_reset_mask]; return }
        after $poll_ms
        set waited [expr {$waited + $poll_ms}]
    }
    error "Timeout waiting for done"
}

set args $argv
if {[llength $args] == 0 || [has_flag args "-h"] || [has_flag args "--help"]} { usage }
set ifm_hex [arg_value args "-ifm" ""]
set wgt_hex [arg_value args "-wgt" ""]
set out_hex [arg_value args "-out" ""]
set ofm_words_s [arg_value args "-ofm-words" ""]
if {$ifm_hex eq "" || $wgt_hex eq "" || $out_hex eq "" || $ofm_words_s eq ""} { usage }

set base_addr [parse_int [arg_value args "-base-addr" "0x70000000"]]
set word_bytes [parse_int [arg_value args "-word-bytes" "8"]]
set ifm_base_word [parse_int [arg_value args "-ifm-base-word" "0x000000"]]
set wgt_base_word [parse_int [arg_value args "-wgt-base-word" "0x08000"]]
set ofm_base_word [parse_int [arg_value args "-ofm-base-word" "0xF0000"]]
set ifm_size_words [parse_int [arg_value args "-ifm-size-words" "0x08000"]]
set wgt_size_words [parse_int [arg_value args "-wgt-size-words" "0x08000"]]
set ofm_size_words [parse_int [arg_value args "-ofm-size-words" "0x08000"]]
set ofm_words [parse_int $ofm_words_s]
set clear_ofm_words [parse_int [arg_value args "-clear-ofm-words" $ofm_words_s]]
set ctrl_mode [arg_value args "-ctrl-mode" "manual"]
set target_index [arg_value args "-target-index" ""]
if {$target_index ne ""} { targets [parse_int $target_index] }
if {$word_bytes <= 0 || [expr {$word_bytes % 4}] != 0} { error "-word-bytes must be a positive multiple of 4" }
if {$ofm_words <= 0} { error "-ofm-words must be > 0" }

set ifm_byte_addr [expr {$base_addr + $ifm_base_word*$word_bytes}]
set wgt_byte_addr [expr {$base_addr + $wgt_base_word*$word_bytes}]
set ofm_byte_addr [expr {$base_addr + $ofm_base_word*$word_bytes}]

log "CNN_V4 KV260 VGG16-like 32 P128/PV8 run"
log "word_bytes=$word_bytes IFM=[format 0x%08X $ifm_byte_addr] WGT=[format 0x%08X $wgt_byte_addr] OFM=[format 0x%08X $ofm_byte_addr]"
if {![has_flag args "-skip-ifm-write"]} { write_ddr_words_from_hex $ifm_hex $ifm_byte_addr $word_bytes $ifm_size_words }
if {![has_flag args "-skip-wgt-write"]} { write_ddr_words_from_hex $wgt_hex $wgt_byte_addr $word_bytes $wgt_size_words }
if {![has_flag args "-skip-ofm-clear"]} { clear_ddr_words $ofm_byte_addr $word_bytes $clear_ofm_words }

if {![has_flag args "-skip-run"]} {
    if {$ctrl_mode eq "manual"} {
        log "Manual mode: start accelerator through VIO/GPIO. Press ENTER after done=1 and error=0."
        gets stdin dummy
    } elseif {$ctrl_mode eq "reg"} {
        set ctrl_base [parse_int [arg_value args "-ctrl-base" "0xA0000000"]]
        start_and_poll_reg $ctrl_base [parse_int [arg_value args "-ctrl-offset" "0x00"]] [parse_int [arg_value args "-status-offset" "0x04"]] [parse_int [arg_value args "-run-mask" "0x1"]] [parse_int [arg_value args "-soft-reset-mask" "0x2"]] [parse_int [arg_value args "-abort-mask" "0x4"]] [parse_int [arg_value args "-done-mask" "0x1"]] [parse_int [arg_value args "-busy-mask" "0x2"]] [parse_int [arg_value args "-error-mask" "0x4"]] [parse_int [arg_value args "-poll-timeout-ms" "60000"]] [parse_int [arg_value args "-poll-interval-ms" "20"]]
    } else { error "Unknown -ctrl-mode $ctrl_mode" }
}
read_ddr_words_to_hex $ofm_byte_addr $word_bytes $ofm_words $out_hex
log "Done. OFM readback: $out_hex"
