head -1 train.csv > tfinal.csv

for filename in $(ls turn*.csv); 
do 
sed 1d $filename >> tfinal.csv;
done;
