#!/bin/bash

# Attention à ne pas ajouter d'espaces après le backslash.
REGEX_EXCLUDE_FILES="/arcane/extras\
|/arcane/src/arcane/utils/internal/json/rapidjson\
|/arcane/src/arcane/dotnet/coreclr/hostfxr\.h\
|/arcane/src/arcane/dotnet/coreclr/coreclr_delegates\.h\
"

CC_H_FILES=$(find $SOURCE -name '*.cc')
CC_H_FILES+=" "
CC_H_FILES+=$(find $SOURCE -name '*.h')

OUTPUT_LOG="Début script de vérification\n\n"

for FILE in $CC_H_FILES;
do

  # On retire les fichiers qui font moins de 10 lignes.
  NB_LINES=$(wc -l < $FILE)
  if (( $NB_LINES < 10 ))
  then
    continue
  fi

  # On retire les fichiers que l'on ne veut pas vérifier.
  COMPT=$(echo "$FILE" | grep -E "$REGEX_EXCLUDE_FILES" | wc -l)
  if (( $COMPT != 0 ))
  then
    continue
  fi


  COPY_LOG=1
  OUTPUT_LOG_FILE=""

  # Vérification du formatage du fichier.
  COMPT=$(file $FILE | grep "UTF-8 Unicode (with BOM)" | wc -l)
  if (( $COMPT == 0 ))
  then
    OUTPUT_LOG_FILE+="  Mauvais format (format nécessaire : UTF-8 with BOM)\n"
    COPY_LOG=0
  fi

  # Vérification du header Emacs.
  COMPT=$(head -1 $FILE | grep -e "-*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-" | wc -l)
  if (( $COMPT == 0 ))
  then
    OUTPUT_LOG_FILE+="  Header Emacs manquant ou mal orthographié\n"
    COPY_LOG=0
  fi


  # On collecte les lignes avec "copyright".
  TEMPO=$(head -30 $FILE | grep -iF "copyright")

  COMPT=$(echo "$TEMPO" | wc -l)
  if (( $COMPT == 0 ))
  then
    OUTPUT_LOG_FILE+="  Copyright manquant\n"
    COPY_LOG=0
  else
    # On vérifie les lignes copyright.
    COMPT=$(echo "$TEMPO" | grep "2000-2022" | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright : Dates manquantes ou non à jours\n"
      COPY_LOG=0
    fi

    COMPT=$(echo "$TEMPO" | grep -i "CEA" | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright : Manque la mention CEA\n"
      COPY_LOG=0
    fi

    COMPT=$(echo "$TEMPO" | grep -i "IFPEN" | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright : Manque la mention IFPEN\n"
      COPY_LOG=0
    fi

    COMPT=$(echo "$TEMPO" | grep -iF "www.cea.fr" | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright : Manque le site web du CEA\n"
      COPY_LOG=0
    fi

    COMPT=$(echo "$TEMPO" | grep -iF "www.ifpenergiesnouvelles.com" | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright : Manque le site web de l'IFPEN\n"
      COPY_LOG=0
    fi

    COMPT=$(echo "$TEMPO" | grep -iF "See the top-level COPYRIGHT file for details." | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright : Manque la position du fichier contenant les détails du Copyright\n"
      COPY_LOG=0
    fi

  fi

  # On vérifie si la licence Apache est précisée.
  COMPT=$(head -30 $FILE | grep "SPDX-License-Identifier: Apache-2.0" | wc -l)
  if (( $COMPT == 0 ))
  then
    OUTPUT_LOG_FILE+="  Licence manquante ou mal orthographiée\n"
    COPY_LOG=0
  fi

  # # On vérifie si les dates sont à jours.
  # COMPT=$(head -30 $FILE | grep "(C) 2000-2022" | wc -l)
  # if (( $COMPT == 0 ))
  # then
  #   OUTPUT_LOG_FILE+="  Dates manquantes ou non à jours\n"
  #   COPY_LOG=0
  # fi

  # S'il y a au moins un problème, on copie dans la variable OUTPUT_LOG.
  if (( $COPY_LOG == 0 ))
  then
    OUTPUT_LOG+="Fichier : $FILE\n$OUTPUT_LOG_FILE\n"
  fi

done

OUTPUT_LOG+="\nFin script de vérification"
echo -e $OUTPUT_LOG