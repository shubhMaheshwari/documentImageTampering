# Testing different methods for document tampering
import cv2
import numpy as np 
import sys
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt


# Load image
image_path = sys.argv[1]
image = cv2.imread(image_path)

# Load xml 
xml_path = sys.argv[2]
tree = ET.parse(xml_path)
root = tree.getroot()

tamperings = []
for elem in root:
	tamperings.appen(elem.attrib())

