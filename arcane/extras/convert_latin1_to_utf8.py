#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import argparse
import codecs

# Chaîne à placer en début de fichier
encode_header = "// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-\n"

def check_has_bom(filename):
    """ Détecte si le fichier 'filename' commence par un BOM et retourne 'true' dans ce cas."""
    raw_f = io.open(filename, "rb")
    utf8_bom = codecs.BOM_UTF8
    raw_first_line = raw_f.read(len(utf8_bom))
    raw_f.close()
    if raw_first_line==utf8_bom:
        return True
    return False

def convert_file(filename):
    # Si le nom de fichier contient la chaîne de caractères 'Generated', considère
    # qu'il s'agit d'un fichier généré et ne fais rien.
    if filename.find("Generated") != (-1):
        print(str.format("File {0} is generated. Skip convertion",filename))
        return False
    file_stat = os.stat(filename)
    file_size = file_stat.st_size
    print(str.format("Filename {0} size={1}", filename,file_size))
    # Si le fichier est trop petit, ne fait rien. Cela signifie qu'il n'est pas
    # utilisé où contient juste des includes
    if file_size<500:
        return False
    # Si le fichier contient le BOM utf8, il est déjà encodé
    if check_has_bom(filename):
        print(str.format("File {0} has utf8-bom",filename))
        return False  
    f = io.open(filename, "r", encoding="iso-8859-1")
    full_content = f.read(file_size)
    f.close()
    # Si le fichier contient la chaîne de caractères 'coding: utf-8',
    # considère que le fichier est déjà encodé en utf-8
    if full_content.find("coding: utf-8") != (-1):
        print(str.format("File {0} already encoded in utf-8",filename))
        return False
    print(str.format("Converting {0} to utf-8",filename))
    utf8_f = open(filename,"w", encoding="utf-8")
    utf8_f.write(codecs.decode(codecs.BOM_UTF8,"utf-8"))
    utf8_f.write(encode_header)
    utf8_f.write(full_content)
    utf8_f.close()
    return True

def exec_convert():
    parser = argparse.ArgumentParser(description='Apply git command to arcane modules.')
    parser.add_argument('files', nargs='+', help='files to convert')
    args = parser.parse_args()
    print(args.files)
    nb_convert = 0;
    converted_files = []
    for x in args.files:
        print(x)
        if convert_file(x):
            nb_convert = nb_convert + 1
            converted_files.append(x)
    print(str.format("NB_CONVERT={0} list={1}",nb_convert,converted_files))

exec_convert()

