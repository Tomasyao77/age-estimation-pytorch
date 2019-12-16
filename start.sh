#! /bin/bash
#项目根目录
basepath="/media/d9lab/data11/tomasyao/workspace/pycharm_ws/age-estimation-pytorch"
DATE=`date +%Y%m%d_%H%M%S`

data_dir=${basepath}/data_dir/morph2
tensorboard=${basepath}/tf_log/morph2
checkpoint=${basepath}/checkpoint/morph2
#log的目录必须存在 上面的目录不存在会自动创建
logs=${basepath}/logs/morph2_all/${DATE}_morph2_all_train_log
logs_test=${basepath}/logs/morph2_all/${DATE}_morph2_all_test_log

if [ $# -ne 1 ] #有且仅有一个参数，否则退出
then
	echo "Usage: /start.sh train[|test|demo]"
	exit 1
else
	echo "starting..."
fi

if [ $1 = "train" ]
then
	#后台运行训练代码
	echo "train..."
	source activate torchg
	cd ${basepath}
#	setsid python ./train.py --data_dir=${data_dir} --tensorboard=${tensorboard} --checkpoint=${checkpoint} > ${logs} 2>&1 &
	setsid python ./train.py > ${logs} 2>&1 &
#	python ./train.py
elif [ $1 = "test" ]
then
	echo "test..."
	source activate torchg
	cd ${basepath}
	#测试速度很快所以就不在后台运行了
	setsid python ./test.py > ${logs_test} 2>&1 &
#	python ./test.py
elif [ $1 = "demo" ]
then
	echo "demo..."
	source activate torch #cpu only
	cd ${basepath}
	python ./demo.py --img_dir=${basepath}/img_dir --output_dir=${basepath}/output_dir --resume=${checkpoint}/epoch079_0.02234_2.2617.pth
elif [ $1 = "tboard" ]
then
    source activate torchg
    cd ${basepath}
	tensorboard --logdir=${tensorboard}
else
	echo "do nothing"
fi