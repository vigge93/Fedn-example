git clone https://github.com/scaleoutsystems/fedn.git
cd fedn

docker-compose build

cd ..

./client_configs/generate_configs.sh 5

