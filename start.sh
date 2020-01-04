#! /bin/bash
#项目根目录
basepath="/media/d9lab/data11/tomasyao/workspace/pycharm_ws/age-estimation-pytorch"
#basepath="/media/zouy/workspace/gitcloneroot/age-estimation-pytorch"
DATE=`date +%Y%m%d_%H%M%S`

data_dir=${basepath}/data_dir/morph2
#0.23, 0.31, 0.33
#tensorboard=${basepath}/tf_log_decay/morph2_align_decay_0.2
tensorboard=${basepath}/tf_log/ceface/2020-1-2
checkpoint=${basepath}/checkpoint/morph2d
#log的目录必须存在 上面的目录不存在会自动创建
logs=${basepath}/logs/morph2_all/${DATE}_morph2_all_train_log
logs_test=${basepath}/logs/morph2_all/${DATE}_morph2_all_test_log
#logs_test=${basepath}/logs/ceface/${DATE}_ceface_test_log
logs_ceface=${basepath}/logs/ceface/${DATE}_ceface_train_log
logs_boxplot=${basepath}/checkpoint/${DATE}_morph2_all_boxplot_log
#单张图片年龄估计
#img_path=${basepath}/data_dir/morph2-align/morph2_align/158175_23M56.jpg
#img_path=${basepath}/data_dir/FG-NET/test/001a02.jpg
img_path=${basepath}/data_dir/CE/110666_514141_9134288_43.0474.jpg
my_resume=${basepath}/checkpoint/morph2_align/epoch079_0.02094_2.6708.pth

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
#	setsid python ./test.py > ${logs_test} 2>&1 &
	python ./test.py
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
elif [ $1 = "age" ] #单张图片年龄估计
then
#    source activate torchg
    cd ${basepath}
	python ./age_estimation.py --img_path=${img_path} --my_resume=${my_resume}
elif [ $1 = "train_ce" ]
then
#后台运行训练代码
	echo "train_ce..."
	source activate torchg
	cd ${basepath}
	setsid python ./train_ce.py > ${logs_ceface} 2>&1 &
#	python ./train_ce.py
elif [ $1 = "boxplot_morph2" ]
then
#后台运行训练代码
	echo "boxplot_morph2..."
	source activate torchg
	cd ${basepath}
	setsid python ./util.py > ${logs_boxplot} 2>&1 &
else
	echo "do nothing"
fi