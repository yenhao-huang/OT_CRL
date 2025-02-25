if [ -z "$1" ]; then
    echo "Usage: $0 <workload_name>"
    exit 1
fi

workload_name=$1

for i in {1..10}; do
    workload="/home/user/Desktop/oss-arch-gym/project_ssd/synthesize_tools/workload/fio/jobfile/${workload_name}.ini"
    fio $workload --output="log/a_${i}.json" --output-format="json"
done

