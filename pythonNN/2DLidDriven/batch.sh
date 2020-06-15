#! /bin/bash

ist1num=10
ist1div=2
ist1step=1

ist2num=1
ist2div=1
ist2step=3


if [ `expr $ist1div + $ist2div` -ne 3 ];then
	echo "error in div"
	exit 1
fi

start=`date +"%s"`
for ((cudanum=0; cudanum<2; cudanum++));do	
	for ((i1=0; i1<$ist1num/$ist1div; i1++)); do
	for ((i2=0; i2<$ist2num/$ist2div; i2++)); do
	        if [ $cudanum -eq 0 ];then
        	        sed -i "s/cuda:1/cuda:0/g" ../tools/NNs/NN.py
			ist1=`expr $i1 + 0`
			ist2=`expr $i2 + 0`
        	        ien1=`expr $ist1 + $ist1step`
	                ien2=`expr $ist2 + $ist2step`			
	        else
        	        sed -i "s/cuda:0/cuda:1/g" ../tools/NNs/NN.py
			if [ $ist1div -eq 2 ]; then
	                        ist1=`expr $i1 + $ist1num / $ist1div`
        	                ist2=`expr $i2`			
			else
                                ist1=`expr $i1`
                                ist2=`expr $i2 + $ist2num / $ist2div`				
			fi
        	        ien1=`expr $ist1 + $ist1step`
	                ien2=`expr $ist2 + $ist2step`			
	        fi	
		echo "$ist1 and $ist2 on cuda $cudanum"
		python3 Cases_test.py $ist1 $ien1 $ist2 $ien2 >$ist1-${ien1}and${ist2}-$ien2.out 2>&1 &
		sleep 10
	done
	done
done
wait
end=`date +"%s"`
echo "the total consumed time =" `expr $end - $start`

echo `expr 4 / 2`
