cd hnd_EK
rm -rf taxonomy
rm -rf datasets

mkdir taxonomy; cd taxonomy
mkdir EK; cd ../

# everything in taxonomy
cp allclasses.txt taxonomy/EK
cp awa_classes_offset_rev1.txt taxonomy/EK
cp taxonomy.npy taxonomy/EK
cp testclasses.txt taxonomy/EK
cp trainvalclasses.txt taxonomy/EK

mkdir datasets; cd datasets
mkdir EK; cd ../

# everything in datasets
cp res101.mat datasets/EK
cp att_splits.mat datasets/EK

cd ../
