aln=$1
jaln=$2

echo "kevin2kevin.py ..."
python2 kevin2kevin.py $aln > tmp_out ; mv tmp_out tmp_in ; cp tmp_in 1;

echo "kevin2jamr.py ..."
python2 kevin2jamr.py tmp_in $jaln > tmp_out ; mv tmp_out tmp_in ; cp tmp_in 2;

echo "fill_kevin_1st.py ..."
python2 fill_kevin_1st.py tmp_in > tmp_out ; mv tmp_out tmp_in ; cp tmp_in 3;

echo "fill_with_head.py ..."
python2 fill_with_head.py tmp_in > tmp_out ; mv tmp_out tmp_in ; cp tmp_in 4;

echo "mrg2.py ..."
python2 mrg2.py tmp_in $jaln > tmp_out ; mv tmp_out tmp_in ; cp tmp_in 5;

echo "fix3.py ..."
python2 fix3.py tmp_in > tmp_out ; mv tmp_out tmp_in ; cp tmp_in 6;

echo "fix2.py ..."
python2 fix2.py tmp_in > tmp_out ; mv tmp_out tmp_in ; cp tmp_in 7;

echo "fill_kevin_1st.py ..."
python2 fill_kevin_1st.py tmp_in > tmp_out ; mv tmp_out tmp_in ; cp tmp_in 8;

echo "fill_with_head.py ..."
python2 fill_with_head.py tmp_in > tmp_out ; mv tmp_out tmp_in ; cp tmp_in 9;

echo "align_the_rest.py ..."
python align_the_rest.py tmp_in > $aln.mrged 

rm tmp_in
