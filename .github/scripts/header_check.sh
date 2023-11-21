#!/bin/bash

# Attention à ne pas ajouter d'espaces après le backslash.
REGEX_EXCLUDE_FILES="/arcane/extras/\
|/arcane/src/arcane/utils/internal/json/rapidjson/\
|/arcane/tutorial/\
|/arcane/src/arcane/packages/\
|/arcane/cmake/test_glibc_malloc_hooks\.cc\
|/arcane/cmake/test_linux_perf_syscall\.cc\
|/arcane/src/arcane/dotnet/coreclr/hostfxr\.h\
|/arcane/src/arcane/dotnet/coreclr/coreclr_delegates\.h\
|/arcane/src/arcane/hyoda/HyodaDbg\.h\
|/arcane/src/arcane/tests/AMR/ErrorEstimate\.h\
|.*Generated.*\
|.*Licensed.*\
"

NUM_FILES_ERROR=0

OUTPUT_LOG="Begin script\n\n"

for FILE in $CC_H_FILES;
do

  if [[ ! -f "$FILE" ]]
  then
    OUTPUT_LOG+="CI Warning -- File not found: $FILE\n"
    continue
  fi

  # On retire les fichiers qui font moins de 10 lignes.
  NB_LINES=$(wc -l < "$FILE")
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
  COMPT=$(file "$FILE" | grep "UTF-8 (with BOM)" | wc -l)
  if (( $COMPT == 0 ))
  then
    OUTPUT_LOG_FILE+="  Bad encoding (need UTF-8 with BOM)\n"
    COPY_LOG=0
  fi



  # Vérification du header Emacs.
  COMPT=$(head -1 "$FILE" | grep -e "-*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-" | wc -l)
  if (( $COMPT == 0 ))
  then
    OUTPUT_LOG_FILE+="  Missing or bad Emacs Header\n"
    COPY_LOG=0
  fi



  # On collecte les lignes avec "copyright".
  TEMPO=$(head -30 "$FILE" | grep -iF "copyright")

  COMPT=$(echo "$TEMPO" | wc -l)
  if (( $COMPT == 0 ))
  then
    OUTPUT_LOG_FILE+="  Missing copyright\n"
    COPY_LOG=0
  else



    # Année de la dernière modification du fichier.
    DATE_FILE=$(git log -1 --pretty="format:%cs" "$FILE")
    DATE_FILE=${DATE_FILE:0:4}

    # Année d'aujourd'hui.
    DATE_TODAY=$(date +"%Y")

    # Récupération des années après "2000-"
    DATE_CR=$(head -30 "$FILE" | grep --color=never -wEo "2000-[0-9]+" | grep --color=never -wEo "[0-9]+$")

    COMPT=$(echo "$DATE_CR" | wc -l)

    # Pas de dates trouvées ou valides.
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright: Missing date\n"
      COPY_LOG=0
    else

      # Vérification des années trouvées.
      OUTPUT_LOG_FILE_DATE=""
      GOOD_DATE=1
      for YEAR in $DATE_CR;
      do
        # On souhaite que les années soient à jours ou correspondent à l'année de dernière modification du fichier.
        # Si au moins une date est bonne, on ne notifie pas d'erreurs dans les logs.
        if (($YEAR != $DATE_TODAY && $YEAR != $DATE_FILE))
        then
          OUTPUT_LOG_FILE_DATE+="  Copyright: Bad year (expected: today (=$DATE_TODAY) or last edit date (=$DATE_FILE) - found: $YEAR)\n"
        else
          GOOD_DATE=0
          break
        fi
      done
      if (( $GOOD_DATE == 1 ))
      then
        OUTPUT_LOG_FILE+=$OUTPUT_LOG_FILE_DATE
        COPY_LOG=0
      fi

    fi



    COMPT=$(echo "$TEMPO" | grep -i "CEA" | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright: Missing CEA mention\n"
      COPY_LOG=0
    fi



    COMPT=$(echo "$TEMPO" | grep -i "IFPEN" | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright: Missing IFPEN mention\n"
      COPY_LOG=0
    fi



    COMPT=$(echo "$TEMPO" | grep -iF "www.cea.fr" | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright: Missing CEA web address\n"
      COPY_LOG=0
    fi



    COMPT=$(echo "$TEMPO" | grep -iF "www.ifpenergiesnouvelles.com" | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright: Missing IFPEN web address\n"
      COPY_LOG=0
    fi



    COMPT=$(echo "$TEMPO" | grep -iF "See the top-level COPYRIGHT file for details." | wc -l)
    if (( $COMPT == 0 ))
    then
      OUTPUT_LOG_FILE+="  Copyright: Missing position of copyright details file\n"
      COPY_LOG=0
    fi

  fi



  # On vérifie si la licence Apache est précisée.
  COMPT=$(head -30 "$FILE" | grep "SPDX-License-Identifier: Apache-2.0" | wc -l)
  if (( $COMPT == 0 ))
  then
    OUTPUT_LOG_FILE+="  Missing or bad licence\n"
    COPY_LOG=0
  fi



  # S'il y a au moins un problème, on copie dans la variable OUTPUT_LOG.
  if (( $COPY_LOG == 0 ))
  then
    OUTPUT_LOG+="File: $FILE\n$OUTPUT_LOG_FILE\n"
    ((NUM_FILES_ERROR++))
  fi



done

OUTPUT_LOG+="\nEnd script"
echo -e $OUTPUT_LOG
exit $NUM_FILES_ERROR
