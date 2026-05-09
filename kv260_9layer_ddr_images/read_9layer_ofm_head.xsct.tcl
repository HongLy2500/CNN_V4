# Read a small head of final OFM region for the 9-layer test.
# For full compare, use a Linux/driver flow or dump a larger memory range.
targets
catch { targets -set -filter {name =~ "PSU"} }

set OFM_PHYS 0x70800000

puts "OFM physical base = $OFM_PHYS"
puts "First 64 x 32-bit words at OFM base:"
mrd $OFM_PHYS 64

puts "Note: each RTL DDR word is 1024 bits = 128 bytes = 32 x 32-bit mrd words."
