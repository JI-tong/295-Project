import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

'''
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
    '''
def xml_to_csv(path, folder):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        folder_name = root.attrib['name']
        for frame in root.iter('frame'):

            for target in frame.find('target_list'):
                num = frame.attrib['num'].zfill(5)
                file_name = folder + '/' + folder_name + '/img' + num + '.jpg'
                ymin = int(round(float(target.find('box').attrib['top'])))
                xmin = int(round(float(target.find('box').attrib['left'])))
                ymax = int(round(float(target.find('box').attrib['height']))) + ymin
                xmax = int(round(float(target.find('box').attrib['width']))) + xmin
                c_type = target.find('attribute').attrib['vehicle_type']
                value = (file_name, xmin, ymin, xmax, ymax, c_type)
                xml_list.append(value)

        print (folder_name + " done")
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xml_df = pd.DataFrame(xml_list, columns=None)
    return xml_df


def main():
    for folder in ['test','train']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path, folder)
        xml_df.to_csv(('images/' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')


main()
