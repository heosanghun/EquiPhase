from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info('transferable-samplers/many-peptides-md')
print(f"License: {info.cardData.get('license', 'Unknown')}")
files = api.list_repo_files('transferable-samplers/many-peptides-md', repo_type='dataset')
print(files)
