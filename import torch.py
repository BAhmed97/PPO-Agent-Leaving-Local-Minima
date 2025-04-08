import torch

# Load the .pth file
file_path = "path_to_your_file/ppo_metadrive_model_exp_num_4.pth"
data = torch.load(file_path, map_location=torch.device('cpu'))

# Check the content type
print(type(data))
if isinstance(data, dict):
    print("Keys:", data.keys())
else:
    print("Data:", data)
