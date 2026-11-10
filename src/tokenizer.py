import re
import json

GUID_REGEX = re.compile(r'\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?')
HEX_REGEX = re.compile(r'0x[0-9a-fA-F]{4,}')
LONG_NUM_REGEX = re.compile(r'\b\d{5,}\b')

KEY_PREFIXES = {
    'accessed_registry_keys': 'REG_ACC:',
    'created_registry_keys': 'REG_CRE:',
    'accessed_files': 'FILE_ACC:',
    'created_files': 'FILE_CRE:',
    'deleted_files': 'FILE_DEL:',
    'changed_files': 'FILE_CHG:',
    'network_traffic_ips': 'NET_IP:',
    'network_dns_lookups': 'DNS:',
    'created_process': 'PROC_CRE:',
    'killed_process': 'PROC_KIL:',
    'processes_tree': 'PROC_TREE:',
    'accessed_mutexes': 'MUTEX_ACC:',
    'created_mutexes': 'MUTEX_CRE:',
    'loaded_modules': 'MOD:',
    'executed_commands': 'CMD:',
    'critical_api_calls': 'API:'
}

def normalize_value(val: str) -> str:
    """
    Standardizes raw behavioral indicators:
    - Replaces dynamic GUIDs, long random hex strings, and process IDs with static tokens.
    - Masks environment-specific username and sample path placeholders.
    - Formats strings into clean, discrete tokens without whitespace.
    """
    val = val.strip().lower()
    val = GUID_REGEX.sub('<guid>', val)
    val = HEX_REGEX.sub('<hex>', val)
    val = LONG_NUM_REGEX.sub('<num>', val)
    val = val.replace('%user%', '<user>').replace('%windir%', '<windir>').replace('%samplepath%', '<sample>')
    val = re.sub(r'\s+', '_', val)
    return val

def extract_tokens_from_json(json_content_bytes: bytes) -> str:
    """
    Parses a RanDS JSON file bytes and returns a space-separated token string
    ready for TF-IDF vectorization.
    """
    data = json.loads(json_content_bytes.decode('utf-8', errors='ignore'))
    tokens = []
    
    for key, prefix in KEY_PREFIXES.items():
        items = data.get(key, [])
        if not items:
            continue
        for item in items:
            if isinstance(item, str):
                norm = normalize_value(item)
                if norm:
                    tokens.append(f"{prefix}{norm}")
            elif isinstance(item, dict):
                for subk, subv in item.items():
                    if isinstance(subv, str):
                        norm = normalize_value(subv)
                        tokens.append(f"{prefix}{subk}_{norm}")
                        
    return " ".join(tokens)
