// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshReaderMng.h                                             (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de lecteurs de maillage.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHREADERMNG_H
#define ARCANE_CORE_MESHREADERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de lecteurs de maillage.
 */
class ARCANE_CORE_EXPORT MeshReaderMng
{
  class Impl;

 public:
 
  MeshReaderMng(ISubDomain* sd);
  MeshReaderMng(const MeshReaderMng&) = delete;
  ~MeshReaderMng();
  const MeshReaderMng& operator=(const MeshReaderMng&) = delete;

 public:

  /*!
   * \brief Lit le maillage dont le nom de fichier est \a file_name.
   *
   * \a file_name doit avoir une extension et le lecteur utilisé est basé
   * sur cette extension.
   * Le maillage créé est associé à un \a IParallelMng séquentiel et aura
   * pour nom \a mesh_name.
   *
   * Cette méthode lève une exception si le maillage ne peut pas être lu.
   */
  IMesh* readMesh(const String& mesh_name,const String& file_name);

  /*!
   * \brief Lit le maillage dont le nom de fichier est \a file_name.
   *
   * \a file_name doit avoir une extension et le lecteur utilisé est basé
   * sur cette extension.
   * Le maillage créé est associé au gestionnaire de parallélisme
   * \a parallel_mng et aura pour nom \a mesh_name.
   *
   * Cette méthode lève une exception si le maillage ne peut pas être lu.
   */
  IMesh* readMesh(const String& mesh_name,const String& file_name,
                  IParallelMng* parallel_mng);

  /*!
   * \brief Si vrai, indique qu'on utilise le système d'unité éventuellement présent
   * dans le format de fichier (\a true par défaut).
   *
   * Cette méthode doit être appelée avant l'appel à readMesh() pour être prise en compte.
   */
  void setUseMeshUnit(bool v);
  //! Indique si on utilise le système d'unité présent dans le fichier
  bool isUseMeshUnit() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

