directory="/scratch/cliao25/imagenet"
if [ ! -d "$directory" ]; then
    echo "copying imagenet to scratch."
    cp ~/cliao25/data/imagenet_subset.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/imagenet_subset.zip -d  /scratch/cliao25/
else
    echo "imagenet exists."
fi



directory="/scratch/cliao25/stanford_cars"
if [ ! -d "$directory" ]; then
    echo "copying stanford_cars to scratch."
    cp ~/cliao25/data/stanford_cars.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/stanford_cars.zip -d  /scratch/cliao25/
else
    echo "stanford_cars exists."
fi



directory="/scratch/cliao25/caltech-101"
if [ ! -d "$directory" ]; then
    echo "copying caltech-101 to scratch."
    cp ~/cliao25/data/caltech-101.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/caltech-101.zip -d  /scratch/cliao25/
else
    echo "caltech-101 exists."
fi



directory="/scratch/cliao25/dtd"
if [ ! -d "$directory" ]; then
    echo "copying dtd to scratch."
    cp ~/cliao25/data/dtd.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/dtd.zip -d  /scratch/cliao25/
else
    echo "dtd exists."
fi



directory="/scratch/cliao25/eurosat"
if [ ! -d "$directory" ]; then
    echo "copying eurosat to scratch."
    cp ~/cliao25/data/eurosat.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/eurosat.zip -d  /scratch/cliao25/
else
    echo "eurosat exists."
fi



directory="/scratch/cliao25/fgvc_aircraft"
if [ ! -d "$directory" ]; then
    echo "copying fgvc_aircraft to scratch."
    cp ~/cliao25/data/fgvc_aircraft.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/fgvc_aircraft.zip -d  /scratch/cliao25/
else
    echo "fgvc_aircraft exists."
fi



directory="/scratch/cliao25/food-101"
if [ ! -d "$directory" ]; then
    echo "copying food-101 to scratch."
    cp ~/cliao25/data/food-101.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/food-101.zip -d  /scratch/cliao25/
else
    echo "food-101 exists."
fi



directory="/scratch/cliao25/oxford_flowers"
if [ ! -d "$directory" ]; then
    echo "copying oxford_flowers to scratch."
    cp ~/cliao25/data/oxford_flowers.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/oxford_flowers.zip -d  /scratch/cliao25/
else
    echo "oxford_flowers exists."
fi



directory="/scratch/cliao25/oxford_pets"
if [ ! -d "$directory" ]; then
    echo "copying oxford_pets to scratch."
    cp ~/cliao25/data/oxford_pets.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/oxford_pets.zip -d  /scratch/cliao25/
else
    echo "oxford_pets exists."
fi



directory="/scratch/cliao25/sun397"
if [ ! -d "$directory" ]; then
    echo "copying sun397 to scratch."
    cp ~/cliao25/data/sun397.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/sun397.zip -d  /scratch/cliao25/
else
    echo "sun397 exists."
fi



directory="/scratch/cliao25/ucf101"
if [ ! -d "$directory" ]; then
    echo "copying ucf101 to scratch."
    cp ~/cliao25/data/ucf101.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/ucf101.zip -d  /scratch/cliao25/
else
    echo "ucf101 exists."
fi

directory="/scratch/cliao25/imagenet-sketch"
if [ ! -d "$directory" ]; then
    echo "copying imagenet-sketch to scratch."
    cp ~/cliao25/data/imagenet-sketch.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/imagenet-sketch.zip -d  /scratch/cliao25/
else
    echo "imagenet-sketch exists."
fi

directory="/scratch/cliao25/imagenet-adversarial"
if [ ! -d "$directory" ]; then
    echo "copying imagenet-adversarial to scratch."
    cp ~/cliao25/data/imagenet-adversarial.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/imagenet-adversarial.zip -d  /scratch/cliao25/
else
    echo "imagenet-adversarial exists."
fi

directory="/scratch/cliao25/imagenet-rendition"
if [ ! -d "$directory" ]; then
    echo "copying imagenet-rendition to scratch."
    cp ~/cliao25/data/imagenet-rendition.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/imagenet-rendition.zip -d  /scratch/cliao25/
else
    echo "imagenet-rendition exists."
fi

directory="/scratch/cliao25/imagenetv2"
if [ ! -d "$directory" ]; then
    echo "copying imagenetv2 to scratch."
    cp ~/cliao25/data/imagenetv2.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/imagenetv2.zip -d  /scratch/cliao25/
else
    echo "imagenetv2 exists."
fi

directory="/scratch/cliao25/imagenet21k_resized"
if [ ! -d "$directory" ]; then
    echo "copying imagenet21k_resized to scratch."
    cp ~/cliao25/data/imagenet21k_resized.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/imagenet21k_resized.zip -d  /scratch/cliao25/
else
    echo "imagenet21k_resized exists."
fi

find /scratch/cliao25 -type f -exec touch {} +