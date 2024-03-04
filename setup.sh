git clone https://github.com/scaleoutsystems/fedn.git
cd fedn

docker-compose build

cd ../client_configs

./generate_configs.sh

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install fedn==0.8.0

docker build -t fedn_cclient .