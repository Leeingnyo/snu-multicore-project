set -e
#  --device gpu/1080
n=`ARGS="../common/examples/input_1.rgb myoutput.rgb" /usr/bin/make run | tail -n 2 | head -n 1 | cut -d ' ' -f 2`
echo $n
sleep 2
watch -n 1 cat "task_$n.stdout"
