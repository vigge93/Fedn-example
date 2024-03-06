for client in $(seq 1 $1);
do
    sed "s/<num>/$client/g" client_template.yaml > client${client}.yaml
done