// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshReader.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service de lecture du maillage.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHREADER_H
#define ARCANE_CORE_IMESHREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface du service gérant la lecture d'un maillage.
 */
class ARCANE_CORE_EXPORT IMeshReader
{
 public:

  //! Types des codes de retour d'une lecture ou écriture
  enum eReturnType
  {
    RTOk, //!< Opération effectuée avec succès
    RTError, //!< Erreur lors de l'opération
    /*! \brief Non concerné par l'opération.
     * Cela signifie que le format de fichier ne correspond
     * pas à ce lecteur ou que le service ne prend pas en compte
     * cette opération.
     */
    RTIrrelevant
  };

 public:

  virtual ~IMeshReader() = default; //!< Libère les ressources

 public:

  //! Vérifie si le service supporte les fichiers avec l'extension \a str
  virtual bool allowExtension(const String& str) = 0;

  /*! \brief Lit un maillage à partir d'un fichier.
   *
   * Lit la géométrie d'un maillage à partir du fichier \a file_name
   * ainsi que les informations de découpage correspondantes
   * et construit le maillage correspondant dans \a mesh.
   *
   * Si \a use_internal_partition est vrai, cela signifie que le partitionnement
   * n'est pas encore fait et qu'il sera fait par Arcane. Dans ce cas,
   * un seul processeur peut lire le maillage. Cependant, les autres
   * doivent tout de même créer tous les groupes possibles.
   * Cet argument n'est utile qu'en parallèle.

   * Si \a dir_name n'est pas nul, ce chemim sert de base pour la lecture
   * des maillages et informations de découpage.
   */
  virtual eReturnType readMeshFromFile(IPrimaryMesh* mesh,
                                       const XmlNode& mesh_element,
                                       const String& file_name,
                                       const String& dir_name,
                                       bool use_internal_partition) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

