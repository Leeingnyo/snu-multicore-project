if [ $# == 1 ]; then
  cat `ls -1 . | grep task | tail -n 1` | wc -l
else
  cat `ls -1 . | grep task | tail -n 1`
fi
