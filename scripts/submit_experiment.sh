#! /bin/sh

options=$(getopt -o ''\
    --long experiment-runner: \
    --long experiment-name: \
    --long storage-url: \
    --long boot-disk-size: \
    --long gpu-type: \
    --long gpu-count: \
    --long no-gpu \
    -- "$@")
[ $? -eq 0 ] || { 
    echo "Incorrect options provided"
    exit 1
}
eval set -- "$options"
storage_url="gs://perfectpitch/experiments/"
boot_disk_size=100
machine_type="n1-standard-4"
gpu_type="nvidia-tesla-p100"
gpu_count="1"
gpu=true
while true; do
    case "$1" in
    --experiment-runner)
        shift
        experiment_runner=$1
        ;;
    --experiment-name)
        shift
        experiment_name=$1
        ;;
    --storage_url)
        shift
        storage_url=$1
        ;;
    --boot-disk-size)
        shift
        boot_disk_size=$1
        ;;
    --machine-type)
        shift
        machine_type=$1
        ;;
    --gpu-type)
        shift
        gpu_type=$1
        ;;
    --gpu-count)
        shift
        gpu_count=$1
        ;;
    --no-gpu)
        gpu=false
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

experiment_dir_url=${storage_url%%/}/$experiment_name
experiment_runner_url=${experiment_dir_url}/runner.sh
experiment_log_url=${experiment_dir_url}/log.txt
experiment_script="#! /bin/bash

gsutil cp -r $experiment_dir_url .
cd $experiment_name
bash runner.sh |& tee runner_log.txt
gsutil cp -r * $experiment_dir_url

name=\$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
zone=\$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')
gcloud --quiet compute instances delete \$name --zone=\$zone
"

instance_name=runner-$experiment_name
image_family="pytorch-1-3-cpu"
if [ $gpu = true ]; then
    image_family="pytorch-1-3-cu100"
fi
accelerator=""
if [ $gpu = true ]; then
    accelerator="type=$gpu_type,count=$gpu_count"
fi
metadata="experiment-script=$experiment_script"
if [ $gpu = ture ]; then
    metadata="install-nvidia-driver=True,"$metadata
fi

# echo experiment_dir_url $experiment_dir_url
# echo experiment_runner_url $experiment_runner_url
# echo experiment_runner_result_url $experiment_runner_result_url
# echo instance_name $instance_name
# echo image_family $image_family
# echo accelerator $accelerator
# echo startup_script $startup_script
# echo metadata $metadata

gsutil cp $experiment_runner $experiment_runner_url
gcloud compute instances create \
    --image-family=$image_family \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=$boot_disk_size \
    --machine-type=$machine_type \
    --accelerator=$accelerator \
    --scopes=storage-rw,compute-rw \
    --metadata="$metadata" \
    $instance_name

cat <<EOFF
run:
gcloud compute ssh $instance_name -- << EOF
curl -H 'Metadata-Flavor: Google' -o experiment_script.sh http://metadata.google.internal/computeMetadata/v1/instance/attributes/experiment-script
chmod u+x experiment_script.sh
tmux new -ds experiment ./experiment_script.sh
EOF
EOFF