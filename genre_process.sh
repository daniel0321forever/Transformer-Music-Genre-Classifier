music_genre="acoustic"

temp_dir="mfcc_${music_genre}_tempdir"
source_dir="mfcc_${music_genre}"
target_dir="mfcc_fig"

echo $temp_dir
[ ! -d $temp_dir ] && mkdir $temp_dir || echo "dir exist"
cp $source_dir/* $temp_dir

# change file name
for file in $temp_dir/*;
do newname="${file/.jpg/|${music_genre}.jpg}";
echo $file
echo $newname;
mv -v "$file" "$newname";
done

# move to final dir, don't move if file exist
mv -vn $temp_dir/* "$target_dir"
rm -r $temp_dir


