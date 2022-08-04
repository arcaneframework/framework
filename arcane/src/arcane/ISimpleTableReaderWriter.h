// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*                                        (C) 2000-2022 */
/*                                                                           */
/* TODO         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLEREADERWRITER_H
#define ARCANE_ISIMPLETABLEREADERWRITER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/ItemTypes.h>
#include <arcane/Directory.h>
#include <arcane/ISimpleTableMng.h>
#include "arcane/IParallelMng.h"
#include "arcane/IMesh.h"
#include <arcane/utils/Iostream.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ARCANE_CORE_EXPORT SimpleTableReaderWriterUtils
{
 public:
  static bool createDirectoryOnlyP0(ISubDomain* sub_domain, Directory& dir)
  {
    int sf = 0;
    if (sub_domain->parallelMng()->commRank() == 0) {
      sf = dir.createDirectory();
    }
    if (sub_domain->parallelMng()->commSize() > 1) {
      sf = sub_domain->parallelMng()->reduce(Parallel::ReduceMax, sf);
    }
    return sf == 0;
  };
  static bool isFileExist(Directory dir, String file)
  {
    std::ifstream stream;
    stream.open(dir.file(file).localstr(), std::ifstream::in);
    bool fin = stream.good();
    stream.close();
    return fin;
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief TODO
 */
class ARCANE_CORE_EXPORT ISimpleTableReaderWriter
{
public:
  virtual ~ISimpleTableReaderWriter() = default;

public:

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'écrire un tableau dans un fichier.
   * L'extension sera ajouté par l'implémentation.
   * 
   * @param dst Le répertoire de destination. Il sera créé s'il n'existe pas.
   * @param file_name Le nom du fichier (sans extension).
   * @return true Si le fichier a bien été écrit.
   * @return false Si le fichier n'a pas pu être écrit.
   */
  virtual bool write(Directory dst, String file_name) = 0;

  /**
   * @brief 
   * TODO : Mettre last_pos à 0
   * @param src 
   * @param file_name 
   * @return true 
   * @return false 
   */
  virtual bool read(Directory src, String file_name) = 0;
  virtual bool clear() = 0;
  virtual void print() = 0;

  /**
   * @brief Méthode permettant de récupérer la précision actuellement
   * utilisée pour l'écriture des valeurs.
   * 
   * @return Integer La précision.
   */
  virtual Integer precision() = 0;
  /**
   * @brief Méthode permettant de modifier la précision du print.
   * 
   * Aussi bien pour la méthode 'print()' que la méthode 'write()'.
   * 
   * \note Un appel à cette méthode sans le paramètre définira la précision
   * par défaut.
   * 
   * @param precision La nouvelle précision.
   */
  virtual void setPrecision(Integer precision = 6) = 0;

  /**
   * @brief Méthode permettant de savoir si le frag 'std::fixed' est
   * actif ou non pour l'écriture des valeurs.
   * 
   * @return true Si oui.
   * @return false Si non.
   */
  virtual bool fixed() = 0;
  /**
   * @brief Méthode permettant de définir le flag 'std::fixed' ou non.
   * 
   * Aussi bien pour la méthode 'print()' que la méthode 'write()'.
   * 
   * Ce flag permet de 'forcer' le nombre de chiffre après la virgule à
   * la précision voulu. Par exemple, si l'on a appelé 'setPrecision(4)',
   * et que l'on appelle 'setFixed(true)', le print de '6.1' donnera '6.1000'.
   * 
   * \note Un appel à cette méthode sans le paramètre définira le flag
   * par défaut.
   * 
   * @param fixed Si le flag 'std::fixed' doit être défini ou non.
   */
  virtual void setFixed(bool fixed = true) = 0;
  
  virtual String typeFile() = 0;

  virtual SimpleTableInternal* internal() = 0;
  virtual void setInternal(SimpleTableInternal* sti) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
