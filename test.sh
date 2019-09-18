test=$1
what=$2

if [ -z "$test" ]
then 
	test="no value"
fi
echo $test
echo $what
echo "please"

python $1
