from fedn import APIClient

client = APIClient(host="localhost", port=8092)
print("Setting package...")
resp = client.set_package("package.tgz", helper="numpyhelper")
print(resp["message"])
print("Setting seed model...")
resp = client.set_initial_model("seed.npz")
print(resp["message"])
