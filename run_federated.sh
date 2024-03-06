num_clients=$1
rounds=$2

if [ $# -lt 2 ]
  then
    echo "Not enough arguments supplied"
	exit
fi

python client/entrypoint.py init_seed
tar -czvhf package.tgz client

docker-compose -f fedn/docker-compose.yaml up --detach

sleep 5

python fedn_upload.py

for client in $(seq 1 $num_clients);
do
	docker run \
      		-v $PWD/client_configs/client${client}.yaml:/app/client.yaml \
      		-e FEDN_CLIENT_ID=$client \
			-e FEDN_CLIENTS=$num_clients \
			--name fedn_client_${client} \
			--network=fedn_default \
			-d \
      		fedn_cclient run client -in client.yaml
done

clients=0
while [ $clients != $num_clients ]
do
	clients=`curl -s http://localhost:8092/list_clients | jq ".count"`
	echo "Clients connected: $clients/$num_clients"
	sleep 1
done

session_id=`uuidgen`
curl -d "{\"session_id\": \"${session_id}\",\"rounds\":${rounds}, \"requested_clients\":${num_clients}, \"min_clients\":${num_clients}}" -H "Content-Type: application/json" -X POST http://localhost:8092/start_session

echo $session_id

status="Started"
while [ "$status" != "\"Finished\"" ]
do
	sleep 10
	status=`curl -s http://localhost:8092/get_session?session_id=${session_id} | jq ".status"`
	curr_rounds=`curl -s http://localhost:8092/list_rounds | jq 'to_entries | max_by(.key|tonumber) | .key|tonumber'`
	echo "Round $curr_rounds/$rounds"
done
echo "Run finished, cleaning up and saving metrics..."

docker rm -f -v $(docker ps -a --filter name=fedn_client* -q)

curl -s localhost:8092/list_validations | jq "to_entries |  map(.value) | .[-1].data | fromjson" > metrics/FED_${num_clients}.json

docker-compose -f fedn/docker-compose.yaml down