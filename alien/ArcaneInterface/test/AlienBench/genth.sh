#!/bin/sh
for s in htsiluth
do
for n in 10 20 40 100 200 400
do
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-1P1TH-${n}.arc ${n} 1 1 32 1
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-1P2TH-${n}.arc ${n} 1 1 32 2
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-1P4TH-${n}.arc ${n} 1 1 32 4
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-1P8TH-${n}.arc ${n} 1 1 32 8
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-1P16TH-${n}.arc ${n} 1 1 32 16
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-1P32TH-${n}.arc ${n} 1 1 32 32
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-2P1TH-${n}.arc ${n} 2 1 16 1
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-2P2TH-${n}.arc ${n} 2 1 16 2
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-2P4TH-${n}.arc ${n} 2 1 16 4
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-2P8TH-${n}.arc ${n} 2 1 16 8
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-2P16TH-${n}.arc ${n} 2 1 16 16
perl alienth.pl AlienBench-${s}.arc.in AlienBench-${s}-2P32TH-${n}.arc ${n} 2 1 32 32
done
done
