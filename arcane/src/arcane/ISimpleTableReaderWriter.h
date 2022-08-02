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
  static bool createDirectoryOnlyP0(IMesh* mesh, Directory& dir)
  {
    int sf = 0;
    if (mesh->parallelMng()->commRank() == 0) {
      sf = dir.createDirectory();
    }
    if (mesh->parallelMng()->commSize() > 1) {
      sf = mesh->parallelMng()->reduce(Parallel::ReduceMax, sf);
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
 * @ingroup StandardService
 * @brief TODO
 */
class ARCANE_CORE_EXPORT ISimpleTableReaderWriter
{
public:
  virtual ~ISimpleTableReaderWriter() = default;

public:

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  virtual bool write(Directory dst, String file) = 0;
  virtual bool read(Directory src, String file) = 0;
  virtual bool clear() = 0;
  virtual void print(Integer only_proc = 0) = 0;

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
   * Aussi bien pour la méthode 'print()' que les méthodes 'writeFile()'.
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
   * Aussi bien pour la méthode 'print()' que les méthodes 'writeFile()'.
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
  virtual void setInternal(SimpleTableInternal& sti) = 0;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
