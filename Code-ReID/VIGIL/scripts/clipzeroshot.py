import os

program_directory = "/data/dzha866/Project/VIGIL/"
os.chdir(program_directory)

gpu = 0

dataset = "Digits"
source_domains = "mnist_m svhn syn"
target_domains = "mnist"

# dataset = "PACS"
# source_domains = "cartoon photo sketch"
# target_domains = "art_painting"

# dataset = "OfficeHome"
# source_domains = "clipart product real_world"
# target_domains = "art"

# dataset = "VLCS"
# source_domains = "labelme pascal sun"
# target_domains = "caltech"

# dataset = "TerraInc"
# source_domains = "location_43 location_46 location_100"
# target_domains = "location_38"

# dataset = "NICO"
# source_domains = "dim grass outdoor rock water"
# target_domains = "autumn"

# dataset = "DomainNet"
# source_domains = "infograph painting quickdraw real sketch"
# target_domains = "clipart"

backbone = "ViTB32"
model = "CLIPZeroShot"
output_dir = "output/" + model + "-" + backbone + "-" + dataset + "-" + target_domains

model_config_file = "config/clipzeroshot.yaml"

command = (
    "python train.py --gpu {} --output-dir {} --dataset {} --source-domains {} --target-domains {} --model {}\
    --model-config-file {}".format(
        gpu,
        output_dir,
        dataset,
        source_domains,
        target_domains,
        model,
        model_config_file,
    )
)
print(command)
os.system(command)
