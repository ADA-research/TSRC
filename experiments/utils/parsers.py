__all__ = ['file_name_raw_data_parser', 'file_name_representation_parser', 'get_data_file_name_parser']

from pathlib import Path


def file_name_raw_data_parser(file_path: Path) -> dict:
    file_name_split = file_path.name.split('_')
    return {
        'dataset_name': file_name_split[0],
        'data_part': file_name_split[1].rstrip('.csv')
    }


def file_name_representation_parser(file_path: Path) -> dict:
    file_name_split = file_path.stem.split('_')  # Use .stem to get the filename without the extension
    dataset_name, model_name, encoder_name = file_name_split

    return {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'encoder_name': encoder_name
    }


def get_data_file_name_parser(data_type: str) -> callable:
    if data_type == 'raw':
        return file_name_raw_data_parser
    elif data_type == 'representation':
        return file_name_representation_parser
    else:
        raise ValueError(f'Invalid data type: {data_type}')
