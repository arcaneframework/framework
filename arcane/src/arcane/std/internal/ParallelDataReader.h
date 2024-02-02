// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelDataReader.h                                        (C) 2000-2024 */
/*                                                                           */
/* Lecteur de IData en parallèle.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_PARALLELDATAREADER_H
#define ARCANE_STD_INTERNAL_PARALLELDATAREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IParallelMng;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture parallèle.
 *
 * Une instance de cette classe est associée à un groupe du maillage.
 *
 * Pour pouvoir l'utiliser, chaque rang du IParallelMng doit spécifier:
 * - la liste des uid qu'il souhaite, à remplir dans wantedUniqueIds()
 * - la liste triée par ordre croissant des uids qui sont gérés par ce rang, à remplir
 * dans writtenUniqueIds().
 * Une fois ceci fait, il faut appeler la méthode sort() pour calculer
 * les infos dont on a besoin pour l'envoie et la réception des valeurs.
 *
 * L'instance est alors utilisable pour toutes les variables qui reposent
 * sur ce groupe et il faut appeler getSortedValues() pour récupérer
 * les valeurs pour une variable.
 * 
 */
class ParallelDataReader
{
  class Impl;

 public:

  explicit ParallelDataReader(IParallelMng* pm);
  ParallelDataReader(const ParallelDataReader& rhs) = delete;
  ~ParallelDataReader();

 public:

  Array<Int64>& writtenUniqueIds();
  Array<Int64>& wantedUniqueIds();
  void sort();
  void getSortedValues(IData* written_data,IData* data);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
