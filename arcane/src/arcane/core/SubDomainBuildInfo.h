// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubDomainBuildInfo.h                                        (C) 2000-2020 */
/*                                                                           */
/* Informations pour construire un sous-domaine.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SUBDOMAINBUILDINFO_H
#define ARCANE_SUBDOMAINBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Paramètres nécessaires à la construction d'un sous-domaine.
 
 Info pour créer un sous-domaine associé au gestionnaire de parallèlisme
 \a parallelMng(), avec le jeu de données de nom \a caseFileName()
 et de contenu \a caseBytes. \a index() est l'indice dans
 la liste des sous-domaines de la session de ce sous-domaine.
 
 Le nom du fichier est purement informatif, et seul le contenu \a caseBytes()
 est utilisé. Si \a caseBytes() n'est pas vide, il doit contenir un
 document XML valide.

 En réplication de domaine, il faut construire l'instance en lui spécifiant
 le IParallelMng correspondant à l'ensemble des sous-domaines et des réplicats,
 qui est allReplicaParallelMng().Sans réplication, cela correspond au gestionnaire
 parallèle standard. 
*/
class ARCANE_CORE_EXPORT SubDomainBuildInfo
{
 public:

  SubDomainBuildInfo(Ref<IParallelMng> pm,Int32 index);
  SubDomainBuildInfo(Ref<IParallelMng> pm,Int32 index,Ref<IParallelMng> all_replica_pm);

 public:

  Ref<IParallelMng> parallelMng() const
  {
    return m_parallel_mng;
  }

  String caseFileName() const { return m_case_file_name; }

  void setCaseFileName(const String& filename)
  {
    m_case_file_name = filename;
  }

  ByteConstArrayView caseBytes() const;
  ByteConstSpan caseContent() const;

  void setCaseBytes(ByteConstArrayView bytes);
  void setCaseContent(ByteConstSpan content);

  Integer index() const { return m_index; }

  Ref<IParallelMng> allReplicaParallelMng() const
  {
    return m_all_replica_parallel_mng;
  }
 
 private:

  Ref<IParallelMng> m_parallel_mng;
  String m_case_file_name; //!< Nom du fichier contenant le jeu de données.
  UniqueArray<std::byte> m_case_content; //!< Contenu du jeu de données
  Int32 m_index; //!< Numéro du sous-domaine dans la session
  //! Gestionnnaire de parallélisme contenant tous les réplica de m_parallel_mng
  Ref<IParallelMng> m_all_replica_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

