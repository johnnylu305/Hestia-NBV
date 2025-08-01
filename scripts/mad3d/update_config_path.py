import os
import yaml
import argparse


def update_path(old_path, old_prefix, new_prefix):
    """
    If old_path starts with old_prefix, replace the prefix with new_prefix.
    Otherwise, return old_path unchanged.
    """
    if old_path.startswith(old_prefix):
        return old_path.replace(old_prefix, new_prefix, 1)
    return old_path


def tuple_constructor(loader, node):
    """
    Constructor to handle YAML !!python/tuple tags.
    """
    return tuple(loader.construct_sequence(node))


def main():
    parser = argparse.ArgumentParser(
        description="Recursively find and update config.yaml files within a directory."
    )
    parser.add_argument(
        "--search_dir",
        required=True,
        help="Path to the directory containing config.yaml files."
    )
    parser.add_argument(
        "--old_prefix",
        default="/home/dsr/Documents/mad3d/New_Dataset20/house3k",
        help="The old prefix to replace in the paths."
    )
    parser.add_argument(
        "--new_prefix",
        default="/home/abc/def/ghi",
        help="The new prefix to replace with in the paths."
    )
    args = parser.parse_args()
    search_dir = args.search_dir
    old_prefix = args.old_prefix
    new_prefix = args.new_prefix
    # Add custom constructor for YAML !!python/tuple
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
    # Recursively traverse the search_dir
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file == "config.yaml":
                config_file = os.path.join(root, file)
                try:
                    # Load the YAML
                    with open(config_file, "r") as f:
                        data = yaml.safe_load(f)
                    # Update the asset_path and usd_dir if present
                    if isinstance(data, dict):  # Ensure the file contains a dictionary
                        if "asset_path" in data:
                            data["asset_path"] = update_path(data["asset_path"], old_prefix, new_prefix)
                        else:
                            print("asset_path not found")
                            exit()
                        if "usd_dir" in data:
                            data["usd_dir"] = update_path(data["usd_dir"], old_prefix, new_prefix)
                        else:
                            print("uds_dir not found")
                            exit()
                        # Write the updated YAML back to the file
                        with open(config_file, "w") as f:
                            yaml.safe_dump(data, f, sort_keys=False)
                        print(f"Updated: {config_file}")
                    else:
                        print(f"Skipping non-dictionary YAML file: {config_file}")
                except Exception as e:
                    print(f"Error processing {config_file}: {e}")


if __name__ == "__main__":
    main()
