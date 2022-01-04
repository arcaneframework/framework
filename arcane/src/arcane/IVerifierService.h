// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVerifierService.h                                          (C) 2000-2013 */
/*                                                                           */
/* Interface du service de vérification des données entre deux exécutions.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IVERIFIERSERVICE_H
#define ARCANE_IVERIFIERSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface du service de vérification des données
 * entre deux exécutions.
 */
class IVerifierService
{
 public:

  //! Libère les ressources
  virtual ~IVerifierService() {}

 public:

  //! Ecrit le fichier référence
  virtual void writeReferenceFile() =0;

  /*! \brief Effectue la vérification à partir du fichier référence.
   *
   * \param parallel_sequential si vrai, indique qu'on compare le résultat
   * d'une exécution parallèle avec celui d'une exécution séquentielle. Cette
   * option est inactive si l'exécution est séquentielle.
   *
   * \param compare_ghost si vrai, indique qu'on compare les résultats aussi
   * sur les entités fantômes. Il est en général normal que les résultats soient
   * différents sur les entités fantômes, car il n'est pas nécessaire que
   * toutes les variables soient synchronisées. C'est pourquoi il vaut mieux
   * en général ne pas faire de vérification sur les entités fantômes. Cette
   * option est inactive si l'exécution est séquentielle.
   */
  virtual void doVerifFromReferenceFile(bool parallel_sequential,bool compare_ghost) =0;

 public:
  
  //! Positionne le nom du fichier contenant les valeurs de référence
  virtual void setFileName(const String& file_name) =0;
  //! Nom du fichier contenant les valeurs de référence
  virtual const String& fileName() const =0;

 public:

  virtual void setResultFileName(const String& file_name) =0;
  virtual const String& resultfileName() const =0;

 public:
  
  //! Positionne le nom du sous répertoire contenant les valeurs de référence
  virtual void setSubDir(const String& sub_dir) =0;
  //! Nom du fichier contenant les valeurs de référence
  virtual const String& subDir() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

