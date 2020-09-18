import xmltodict
from dicttoxml import dicttoxml


class XmlHandler:
    @staticmethod
    def load_param(fpath):
        with open(fpath, 'rt') as f:
            xml = f.read()
        dict = xmltodict.parse(xml)['root']
        if 'bool' in dict:
            for key, value in dict['bool'].items():
                value = XmlHandler.str_to_bool(value)
                dict['bool'][key] = value
        if 'numeric' in dict:
            for key, value in dict['numeric'].items():
                value = float(value)
                dict['numeric'][key] = value
        return dict

    @staticmethod
    def to_xml(dict):
        xml = dicttoxml(dict, attr_type=False)
        with open('parameter.xml', 'wt') as f:
            f.write(parseString(xml).toprettyxml())
