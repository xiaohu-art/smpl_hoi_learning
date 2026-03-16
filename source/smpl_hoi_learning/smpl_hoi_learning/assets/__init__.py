import os

# Conveniences to other module directories via relative paths
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))

# MJCF file path
# SUB10_XML_PATH = os.path.join(ASSET_DIR, "sub10.xml")
SUB10_XML_PATH = os.path.join(ASSET_DIR, "sub10_split.xml")
SUB16_XML_PATH = os.path.join(ASSET_DIR, "sub16_split.xml")
