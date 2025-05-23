from typing import Optional

METADATA =\
{
	'name': 'TMA Synthetics - Face Services',
	# 'description': 'Industry leading face manipulation platform',
	'version': '1.2',
	# 'license': 'MIT',
	# 'author': 'TMA Synthetics',
	# 'url': 'https://facefusion.io'  
}


def get(key : str) -> Optional[str]:
	if key in METADATA:
		return METADATA.get(key)
	return None
