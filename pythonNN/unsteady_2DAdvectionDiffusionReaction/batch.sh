#! /bin/bash

ist1num=6
ist2num=2
ist3num=1
ist4num=5
ist5num=5
resultsdir="resultsNew_LossWeight"

for ((i1=0; i1<$ist1num; i1++)); do
for ((i2=0; i2<$ist2num; i2++)); do
for ((i3=0; i3<$ist3num; i3++)); do
for ((i4=0; i4<$ist4num; i4++)); do
for ((i5=0; i5<$ist5num; i5++)); do
	ist1=`expr ${ist1num} - ${i1} - 1`
	ien1=`expr ${ist1} + 1`
	ist2=`expr ${ist2num} - ${i2} - 1`
	ien2=`expr ${ist2} + 1`
        ist3=`expr ${ist3num} - ${i3} - 1`
        ien3=`expr ${ist3} + 1`
        ist4=`expr ${ist4num} - ${i4} - 1`
        ien4=`expr ${ist4} + 1`
        ist5=`expr ${ist5num} - ${i5} - 1`
        ien5=`expr ${ist5} + 1`
	sbatch run.sh "python3 Cases_test.py $ist1 $ien1 $ist2 $ien2 $ist3 $ien3 $ist4 $ien4 $ist5 $ien5  ${resultsdir}" "log_${resultsdir}_${ist1}-${ien1}and${ist2}-${ien2}and${ist3}-${ien3}and${ist4}-${ien4}and${ist5}-${ien5}"
#	sleep 1
done
done
done
done
done
