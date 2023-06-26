#!/bin/sh
for s in trilinosmuelu
do
#for n in 10 20 40 100 200 400
for n in 10 
do
perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-1P-${n}.arc ${n} 1 1
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-2P-${n}.arc ${n} 2 1
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-4P-${n}.arc ${n} 2 2
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-8P-${n}.arc ${n} 4 2
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-16P-${n}.arc ${n} 4 4
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-32P-${n}.arc ${n} 8 4
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-64P-${n}.arc ${n} 8 8
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-128P-${n}.arc ${n} 16 8
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-256P-${n}.arc ${n} 16 16
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-512P-${n}.arc ${n} 32 16
#perl alien.pl AlienBench-${s}.arc.in AlienBench-${s}-1024P-${n}.arc ${n} 32 32
done
done
