from data.processing import transforms
import data.processing.augmennt as augmennt
import yaml

special_transforms_list = [
    'Compose', 'RandomApply', 'RandomOrder',
    'RandomOrder', 'RandomChoice',
]

def parse_transform_pipeline(option_transform:dict|str) -> list:
    transform_list = []
    
    if isinstance(option_transform, str):
        option_path = option_transform
        with open(option_path, 'r') as file_option:
            option_transform = yaml.safe_load(file_option)

    for transform_name in option_transform:
        #print(transform_name)
        if transform_name not in augmennt.__all__:
            raise NotImplementedError(f'tranform [{transform_name}] ' \
            + f'is not included in [{augmennt.__all__}]')

        params:dict = option_transform[transform_name] 
    
        if transform_name in special_transforms_list:
            params['transforms'] = parse_transform_pipeline(params['transforms'])

        transform_list += [augmennt.__dict__[transform_name](**params)]
        
    return transform_list