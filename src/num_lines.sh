find -type f -name \*.py -exec wc -l {} \;       | cut -d' ' -f1       | while read line; do           res=$(($res + $line));           echo $res;         done       | tail -1