# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "AXI_ADDR_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "AXI_DATA_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "AXI_DDR_BASE_ADDR" -parent ${Page_0}
  ipgui::add_param $IPINST -name "AXI_ID_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CFG_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DATA_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DDR_ADDR_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DDR_WORD_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "F_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "HT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "H_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "K_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OFM_BANK_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OFM_LINEAR_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PC_MODE2" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PF_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PF_MODE2" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PSUM_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PTOTAL" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PV_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WGT_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WR_FIFO_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "W_MAX" -parent ${Page_0}


}

proc update_PARAM_VALUE.AXI_ADDR_W { PARAM_VALUE.AXI_ADDR_W } {
	# Procedure called to update AXI_ADDR_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AXI_ADDR_W { PARAM_VALUE.AXI_ADDR_W } {
	# Procedure called to validate AXI_ADDR_W
	return true
}

proc update_PARAM_VALUE.AXI_DATA_W { PARAM_VALUE.AXI_DATA_W } {
	# Procedure called to update AXI_DATA_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AXI_DATA_W { PARAM_VALUE.AXI_DATA_W } {
	# Procedure called to validate AXI_DATA_W
	return true
}

proc update_PARAM_VALUE.AXI_DDR_BASE_ADDR { PARAM_VALUE.AXI_DDR_BASE_ADDR } {
	# Procedure called to update AXI_DDR_BASE_ADDR when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AXI_DDR_BASE_ADDR { PARAM_VALUE.AXI_DDR_BASE_ADDR } {
	# Procedure called to validate AXI_DDR_BASE_ADDR
	return true
}

proc update_PARAM_VALUE.AXI_ID_W { PARAM_VALUE.AXI_ID_W } {
	# Procedure called to update AXI_ID_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AXI_ID_W { PARAM_VALUE.AXI_ID_W } {
	# Procedure called to validate AXI_ID_W
	return true
}

proc update_PARAM_VALUE.CFG_DEPTH { PARAM_VALUE.CFG_DEPTH } {
	# Procedure called to update CFG_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CFG_DEPTH { PARAM_VALUE.CFG_DEPTH } {
	# Procedure called to validate CFG_DEPTH
	return true
}

proc update_PARAM_VALUE.C_MAX { PARAM_VALUE.C_MAX } {
	# Procedure called to update C_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_MAX { PARAM_VALUE.C_MAX } {
	# Procedure called to validate C_MAX
	return true
}

proc update_PARAM_VALUE.DATA_W { PARAM_VALUE.DATA_W } {
	# Procedure called to update DATA_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DATA_W { PARAM_VALUE.DATA_W } {
	# Procedure called to validate DATA_W
	return true
}

proc update_PARAM_VALUE.DDR_ADDR_W { PARAM_VALUE.DDR_ADDR_W } {
	# Procedure called to update DDR_ADDR_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DDR_ADDR_W { PARAM_VALUE.DDR_ADDR_W } {
	# Procedure called to validate DDR_ADDR_W
	return true
}

proc update_PARAM_VALUE.DDR_WORD_W { PARAM_VALUE.DDR_WORD_W } {
	# Procedure called to update DDR_WORD_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DDR_WORD_W { PARAM_VALUE.DDR_WORD_W } {
	# Procedure called to validate DDR_WORD_W
	return true
}

proc update_PARAM_VALUE.F_MAX { PARAM_VALUE.F_MAX } {
	# Procedure called to update F_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.F_MAX { PARAM_VALUE.F_MAX } {
	# Procedure called to validate F_MAX
	return true
}

proc update_PARAM_VALUE.HT { PARAM_VALUE.HT } {
	# Procedure called to update HT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.HT { PARAM_VALUE.HT } {
	# Procedure called to validate HT
	return true
}

proc update_PARAM_VALUE.H_MAX { PARAM_VALUE.H_MAX } {
	# Procedure called to update H_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.H_MAX { PARAM_VALUE.H_MAX } {
	# Procedure called to validate H_MAX
	return true
}

proc update_PARAM_VALUE.K_MAX { PARAM_VALUE.K_MAX } {
	# Procedure called to update K_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.K_MAX { PARAM_VALUE.K_MAX } {
	# Procedure called to validate K_MAX
	return true
}

proc update_PARAM_VALUE.OFM_BANK_DEPTH { PARAM_VALUE.OFM_BANK_DEPTH } {
	# Procedure called to update OFM_BANK_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OFM_BANK_DEPTH { PARAM_VALUE.OFM_BANK_DEPTH } {
	# Procedure called to validate OFM_BANK_DEPTH
	return true
}

proc update_PARAM_VALUE.OFM_LINEAR_DEPTH { PARAM_VALUE.OFM_LINEAR_DEPTH } {
	# Procedure called to update OFM_LINEAR_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OFM_LINEAR_DEPTH { PARAM_VALUE.OFM_LINEAR_DEPTH } {
	# Procedure called to validate OFM_LINEAR_DEPTH
	return true
}

proc update_PARAM_VALUE.PC_MODE2 { PARAM_VALUE.PC_MODE2 } {
	# Procedure called to update PC_MODE2 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PC_MODE2 { PARAM_VALUE.PC_MODE2 } {
	# Procedure called to validate PC_MODE2
	return true
}

proc update_PARAM_VALUE.PF_MAX { PARAM_VALUE.PF_MAX } {
	# Procedure called to update PF_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PF_MAX { PARAM_VALUE.PF_MAX } {
	# Procedure called to validate PF_MAX
	return true
}

proc update_PARAM_VALUE.PF_MODE2 { PARAM_VALUE.PF_MODE2 } {
	# Procedure called to update PF_MODE2 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PF_MODE2 { PARAM_VALUE.PF_MODE2 } {
	# Procedure called to validate PF_MODE2
	return true
}

proc update_PARAM_VALUE.PSUM_W { PARAM_VALUE.PSUM_W } {
	# Procedure called to update PSUM_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PSUM_W { PARAM_VALUE.PSUM_W } {
	# Procedure called to validate PSUM_W
	return true
}

proc update_PARAM_VALUE.PTOTAL { PARAM_VALUE.PTOTAL } {
	# Procedure called to update PTOTAL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PTOTAL { PARAM_VALUE.PTOTAL } {
	# Procedure called to validate PTOTAL
	return true
}

proc update_PARAM_VALUE.PV_MAX { PARAM_VALUE.PV_MAX } {
	# Procedure called to update PV_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PV_MAX { PARAM_VALUE.PV_MAX } {
	# Procedure called to validate PV_MAX
	return true
}

proc update_PARAM_VALUE.WGT_DEPTH { PARAM_VALUE.WGT_DEPTH } {
	# Procedure called to update WGT_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WGT_DEPTH { PARAM_VALUE.WGT_DEPTH } {
	# Procedure called to validate WGT_DEPTH
	return true
}

proc update_PARAM_VALUE.WR_FIFO_DEPTH { PARAM_VALUE.WR_FIFO_DEPTH } {
	# Procedure called to update WR_FIFO_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WR_FIFO_DEPTH { PARAM_VALUE.WR_FIFO_DEPTH } {
	# Procedure called to validate WR_FIFO_DEPTH
	return true
}

proc update_PARAM_VALUE.W_MAX { PARAM_VALUE.W_MAX } {
	# Procedure called to update W_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.W_MAX { PARAM_VALUE.W_MAX } {
	# Procedure called to validate W_MAX
	return true
}


proc update_MODELPARAM_VALUE.DATA_W { MODELPARAM_VALUE.DATA_W PARAM_VALUE.DATA_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DATA_W}] ${MODELPARAM_VALUE.DATA_W}
}

proc update_MODELPARAM_VALUE.PSUM_W { MODELPARAM_VALUE.PSUM_W PARAM_VALUE.PSUM_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PSUM_W}] ${MODELPARAM_VALUE.PSUM_W}
}

proc update_MODELPARAM_VALUE.PTOTAL { MODELPARAM_VALUE.PTOTAL PARAM_VALUE.PTOTAL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PTOTAL}] ${MODELPARAM_VALUE.PTOTAL}
}

proc update_MODELPARAM_VALUE.PV_MAX { MODELPARAM_VALUE.PV_MAX PARAM_VALUE.PV_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PV_MAX}] ${MODELPARAM_VALUE.PV_MAX}
}

proc update_MODELPARAM_VALUE.PF_MAX { MODELPARAM_VALUE.PF_MAX PARAM_VALUE.PF_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PF_MAX}] ${MODELPARAM_VALUE.PF_MAX}
}

proc update_MODELPARAM_VALUE.PC_MODE2 { MODELPARAM_VALUE.PC_MODE2 PARAM_VALUE.PC_MODE2 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PC_MODE2}] ${MODELPARAM_VALUE.PC_MODE2}
}

proc update_MODELPARAM_VALUE.PF_MODE2 { MODELPARAM_VALUE.PF_MODE2 PARAM_VALUE.PF_MODE2 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PF_MODE2}] ${MODELPARAM_VALUE.PF_MODE2}
}

proc update_MODELPARAM_VALUE.C_MAX { MODELPARAM_VALUE.C_MAX PARAM_VALUE.C_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_MAX}] ${MODELPARAM_VALUE.C_MAX}
}

proc update_MODELPARAM_VALUE.F_MAX { MODELPARAM_VALUE.F_MAX PARAM_VALUE.F_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.F_MAX}] ${MODELPARAM_VALUE.F_MAX}
}

proc update_MODELPARAM_VALUE.W_MAX { MODELPARAM_VALUE.W_MAX PARAM_VALUE.W_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.W_MAX}] ${MODELPARAM_VALUE.W_MAX}
}

proc update_MODELPARAM_VALUE.H_MAX { MODELPARAM_VALUE.H_MAX PARAM_VALUE.H_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.H_MAX}] ${MODELPARAM_VALUE.H_MAX}
}

proc update_MODELPARAM_VALUE.HT { MODELPARAM_VALUE.HT PARAM_VALUE.HT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.HT}] ${MODELPARAM_VALUE.HT}
}

proc update_MODELPARAM_VALUE.K_MAX { MODELPARAM_VALUE.K_MAX PARAM_VALUE.K_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.K_MAX}] ${MODELPARAM_VALUE.K_MAX}
}

proc update_MODELPARAM_VALUE.WGT_DEPTH { MODELPARAM_VALUE.WGT_DEPTH PARAM_VALUE.WGT_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WGT_DEPTH}] ${MODELPARAM_VALUE.WGT_DEPTH}
}

proc update_MODELPARAM_VALUE.OFM_BANK_DEPTH { MODELPARAM_VALUE.OFM_BANK_DEPTH PARAM_VALUE.OFM_BANK_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OFM_BANK_DEPTH}] ${MODELPARAM_VALUE.OFM_BANK_DEPTH}
}

proc update_MODELPARAM_VALUE.OFM_LINEAR_DEPTH { MODELPARAM_VALUE.OFM_LINEAR_DEPTH PARAM_VALUE.OFM_LINEAR_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OFM_LINEAR_DEPTH}] ${MODELPARAM_VALUE.OFM_LINEAR_DEPTH}
}

proc update_MODELPARAM_VALUE.CFG_DEPTH { MODELPARAM_VALUE.CFG_DEPTH PARAM_VALUE.CFG_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CFG_DEPTH}] ${MODELPARAM_VALUE.CFG_DEPTH}
}

proc update_MODELPARAM_VALUE.DDR_ADDR_W { MODELPARAM_VALUE.DDR_ADDR_W PARAM_VALUE.DDR_ADDR_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DDR_ADDR_W}] ${MODELPARAM_VALUE.DDR_ADDR_W}
}

proc update_MODELPARAM_VALUE.DDR_WORD_W { MODELPARAM_VALUE.DDR_WORD_W PARAM_VALUE.DDR_WORD_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DDR_WORD_W}] ${MODELPARAM_VALUE.DDR_WORD_W}
}

proc update_MODELPARAM_VALUE.AXI_ADDR_W { MODELPARAM_VALUE.AXI_ADDR_W PARAM_VALUE.AXI_ADDR_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXI_ADDR_W}] ${MODELPARAM_VALUE.AXI_ADDR_W}
}

proc update_MODELPARAM_VALUE.AXI_DATA_W { MODELPARAM_VALUE.AXI_DATA_W PARAM_VALUE.AXI_DATA_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXI_DATA_W}] ${MODELPARAM_VALUE.AXI_DATA_W}
}

proc update_MODELPARAM_VALUE.AXI_ID_W { MODELPARAM_VALUE.AXI_ID_W PARAM_VALUE.AXI_ID_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXI_ID_W}] ${MODELPARAM_VALUE.AXI_ID_W}
}

proc update_MODELPARAM_VALUE.AXI_DDR_BASE_ADDR { MODELPARAM_VALUE.AXI_DDR_BASE_ADDR PARAM_VALUE.AXI_DDR_BASE_ADDR } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXI_DDR_BASE_ADDR}] ${MODELPARAM_VALUE.AXI_DDR_BASE_ADDR}
}

proc update_MODELPARAM_VALUE.WR_FIFO_DEPTH { MODELPARAM_VALUE.WR_FIFO_DEPTH PARAM_VALUE.WR_FIFO_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WR_FIFO_DEPTH}] ${MODELPARAM_VALUE.WR_FIFO_DEPTH}
}

