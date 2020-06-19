set -e
#  --device gpu/1080
if [ $# == 1 ]; then
  n=$1
else
  n=1
fi
echo "../common/examples/input_${n}.rgb myoutput.rgb"
n=`ARGS="../common/examples/input_${n}.rgb myoutput.rgb" /usr/bin/make run | tail -n 2 | head -n 1 | cut -d ' ' -f 2`
echo $n
sleep 2
watch -n 1 cat "task_$n.stdout"
